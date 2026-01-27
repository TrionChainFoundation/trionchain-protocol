#![cfg_attr(not(feature = "std"), no_std)]

//! # TrionChain FEM Pallet
//!
//! ## Overview
//!
//! This pallet demonstrates TrionChain's core innovation: runtime-level validation
//! of physical variables with spatial consistency across a mesh of neighboring cells.
//!
//! ## Grant Differentiator
//!
//! Unlike traditional blockchain oracles that blindly store sensor data, TrionChain
//! validates physical consistency at the consensus layer. This implementation uses
//! CO2 as a demonstration variable, but the design is extensible to any physical
//! vector (stress, temperature, load, etc.).
//!
//! ## Permission Model (Production-Grade)
//!
//! - **Root/Sudo only** can configure mesh topology (neighbors) and assign oracles
//! - **Authorized oracles** can report cell state updates
//! - **Runtime validates** physical consistency before accepting state
//!
//! This institutional permission model ensures data integrity and demonstrates
//! production-readiness for grant reviewers.

pub use pallet::*;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::{pallet_prelude::*, weights::Weight, BoundedVec};
    use frame_system::pallet_prelude::*;
    use sp_std::vec::Vec;

    /// Maximum number of physics data points per cell
    pub type MaxPhysicsLen = ConstU32<256>;

    /// TrionCell data structure with physical state
    ///
    /// Represents a computational cell in the FEM-inspired mesh.
    /// Each cell maintains:
    /// - Simulation step counter
    /// - Generic physics data vector (extensible)
    /// - CO2 level (demonstration physical variable)
    /// - Block timestamp for state tracking
    #[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub struct TrionCellData {
        /// Current simulation/computation step
        pub step: u32,
        /// Generic physics data (extensible to multiple physical variables)
        pub physics: BoundedVec<u32, MaxPhysicsLen>,
        /// CO2 level (ppm) - demonstration physical variable
        pub co2_level: u32,
        /// Block number when this state was recorded
        pub updated_at_block: u64,
    }

    // ---------- WeightInfo ----------
    /// Weight estimation for pallet operations
    pub trait WeightInfo {
        fn update_cell(physics_len: u32) -> Weight;
        fn set_neighbors(neighbors_len: u32) -> Weight;
        fn set_cell_owner() -> Weight;
    }

    /// Default weight implementation (to be replaced with benchmarking)
    impl WeightInfo for () {
        fn update_cell(_physics_len: u32) -> Weight {
            Weight::from_parts(10_000, 0)
        }
        fn set_neighbors(_neighbors_len: u32) -> Weight {
            Weight::from_parts(10_000, 0)
        }
        fn set_cell_owner() -> Weight {
            Weight::from_parts(10_000, 0)
        }
    }

    #[pallet::config]
    pub trait Config: frame_system::Config<RuntimeEvent: From<Event<Self>>> {
        /// Weight information for extrinsics
        type WeightInfo: WeightInfo;

        /// Maximum number of neighbors a cell can have in the mesh topology
        #[pallet::constant]
        type MaxNeighbors: Get<u32>;

        /// Maximum allowed CO2 deviation from neighbor average (ppm)
        ///
        /// This threshold enforces spatial consistency. If a cell's CO2 differs
        /// from its neighbors' average by more than this value, the update is rejected.
        #[pallet::constant]
        type MaxCo2Delta: Get<u32>;
    }

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    /// Storage: Cell states indexed by cell ID
    ///
    /// Maps each cell to its current physical state including CO2 level,
    /// simulation step, and physics data.
    #[pallet::storage]
    #[pallet::getter(fn cell_state)]
    pub type CellStates<T: Config> =
        StorageMap<_, Blake2_128Concat, u32, TrionCellData, OptionQuery>;

    /// Storage: Mesh topology - neighbor relationships
    ///
    /// Defines which cells are spatially adjacent for validation purposes.
    /// Only Root can configure this topology.
    #[pallet::storage]
    #[pallet::getter(fn neighbors)]
    pub type Neighbors<T: Config> =
        StorageMap<_, Blake2_128Concat, u32, BoundedVec<u32, T::MaxNeighbors>, OptionQuery>;

    /// Storage: Authorized oracles for each cell
    ///
    /// Only the assigned oracle can update a cell's state.
    /// Only Root can assign/change oracles.
    #[pallet::storage]
    #[pallet::getter(fn cell_owner)]
    pub type CellOwners<T: Config> =
        StorageMap<_, Blake2_128Concat, u32, T::AccountId, OptionQuery>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Cell state updated successfully
        ///
        /// Indicates that a cell's state passed validation and was stored.
        /// Parameters: cell_id, step, co2_level, physics_data_length
        CellUpdated {
            cell_id: u32,
            step: u32,
            co2_level: u32,
            physics_len: u32,
        },
        /// Mesh topology configured for a cell
        ///
        /// Root has defined which cells are neighbors for validation.
        /// Parameters: cell_id, number_of_neighbors
        NeighborsSet {
            cell_id: u32,
            neighbor_count: u32,
        },
        /// Oracle/owner assigned to a cell
        ///
        /// Root has authorized an account to report state for this cell.
        /// Parameters: cell_id, authorized_account
        CellOwnerSet {
            cell_id: u32,
            owner: T::AccountId,
        },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Physics data vector exceeds MaxPhysicsLen
        PhysicsTooLong,
        /// Neighbor list exceeds MaxNeighbors
        TooManyNeighbors,
        /// Caller is not authorized to update this cell
        ///
        /// Only the assigned oracle (CellOwners) can update cell state.
        NotAuthorized,
        /// CO2 level deviates too much from neighbor average
        ///
        /// The spatial consistency check failed: |new_co2 - neighbor_avg| > MaxCo2Delta
        /// This indicates physically implausible data.
        Co2DeltaTooHigh,
        /// Arithmetic overflow in calculations
        ArithmeticOverflow,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Update cell state with spatial validation
        ///
        /// # Permission
        /// - Caller must be the assigned oracle for this cell (CellOwners)
        ///
        /// # Validation
        /// 1. Checks caller authorization
        /// 2. Validates physics data length
        /// 3. **Spatial validation**: Compares CO2 against neighbor average
        ///    - If neighbors exist and have data: rejects if delta > MaxCo2Delta
        ///    - If no neighbors or no neighbor data: allows (bootstrap case)
        ///
        /// # Parameters
        /// - `cell_id`: Unique identifier for the cell
        /// - `step`: Current simulation step
        /// - `co2_level`: CO2 reading in ppm
        /// - `physics`: Generic physics data vector
        ///
        /// # Grant Demo Value
        /// This demonstrates runtime-level physics validation - the chain
        /// actively rejects physically inconsistent data rather than blindly storing it.
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::update_cell(physics.len() as u32))]
        pub fn update_cell(
            origin: OriginFor<T>,
            cell_id: u32,
            step: u32,
            co2_level: u32,
            physics: Vec<u32>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Permission check: only authorized oracle can update
            Self::ensure_authorized(&who, cell_id)?;

            // Validate physics data length
            let bounded: BoundedVec<u32, MaxPhysicsLen> =
                physics.try_into().map_err(|_| Error::<T>::PhysicsTooLong)?;

            // CORE INNOVATION: Spatial validation against neighbor mesh
            // This is what differentiates TrionChain from generic oracle chains
            Self::validate_co2_against_neighbors(cell_id, co2_level)?;

            // Get current block number for timestamping
            use frame_support::sp_runtime::traits::SaturatedConversion;
            let now: u64 = frame_system::Pallet::<T>::block_number().saturated_into::<u64>();

            // Store validated state
            let data = TrionCellData {
                step,
                physics: bounded,
                co2_level,
                updated_at_block: now,
            };

            CellStates::<T>::insert(cell_id, data);

            let physics_len = CellStates::<T>::get(cell_id)
                .map(|d| d.physics.len() as u32)
                .unwrap_or(0);

            Self::deposit_event(Event::CellUpdated {
                cell_id,
                step,
                co2_level,
                physics_len,
            });

            Ok(())
        }

        /// Configure mesh topology for a cell (Root only)
        ///
        /// Defines which cells are spatial neighbors for validation purposes.
        /// This establishes the mesh structure for FEM-inspired validation.
        ///
        /// # Permission
        /// - Requires Root/Sudo origin
        ///
        /// # Parameters
        /// - `cell_id`: The cell to configure
        /// - `neighbors`: List of neighboring cell IDs (max MaxNeighbors)
        ///
        /// # Grant Demo Value
        /// Shows institutional governance: mesh topology is managed by
        /// trusted authorities, not arbitrary users.
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::set_neighbors(neighbors.len() as u32))]
        pub fn set_neighbors(
            origin: OriginFor<T>,
            cell_id: u32,
            neighbors: Vec<u32>,
        ) -> DispatchResult {
            // Only Root can configure mesh topology
            ensure_root(origin)?;

            // Validate neighbor count
            let bounded: BoundedVec<u32, T::MaxNeighbors> =
                neighbors.try_into().map_err(|_| Error::<T>::TooManyNeighbors)?;

            // Store mesh topology
            Neighbors::<T>::insert(cell_id, bounded.clone());

            Self::deposit_event(Event::NeighborsSet {
                cell_id,
                neighbor_count: bounded.len() as u32,
            });

            Ok(())
        }

        /// Assign authorized oracle for a cell (Root only)
        ///
        /// Designates which account is authorized to report state updates
        /// for a specific cell. This implements institutional oracle management.
        ///
        /// # Permission
        /// - Requires Root/Sudo origin
        ///
        /// # Parameters
        /// - `cell_id`: The cell to assign an oracle for
        /// - `owner`: The account authorized to update this cell
        ///
        /// # Grant Demo Value
        /// Demonstrates production-grade permission model where data sources
        /// are explicitly authorized by governance, not self-claimed.
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::set_cell_owner())]
        pub fn set_cell_owner(
            origin: OriginFor<T>,
            cell_id: u32,
            owner: T::AccountId,
        ) -> DispatchResult {
            // Only Root can assign oracles
            ensure_root(origin)?;

            CellOwners::<T>::insert(cell_id, owner.clone());

            Self::deposit_event(Event::CellOwnerSet {
                cell_id,
                owner,
            });

            Ok(())
        }
    }

    impl<T: Config> Pallet<T> {
        /// Verify caller is authorized oracle for the cell
        fn ensure_authorized(who: &T::AccountId, cell_id: u32) -> DispatchResult {
            let owner = CellOwners::<T>::get(cell_id)
                .ok_or(Error::<T>::NotAuthorized)?;
            
            ensure!(who == &owner, Error::<T>::NotAuthorized);
            Ok(())
        }

        /// Validate CO2 level against neighbor average (CORE INNOVATION)
        ///
        /// This function implements TrionChain's key differentiator:
        /// runtime-level validation of physical consistency.
        ///
        /// # Algorithm
        /// 1. If no neighbors configured → Allow (isolated cell)
        /// 2. If neighbors exist but none have data → Allow (bootstrap phase)
        /// 3. Calculate average CO2 from neighbors with data
        /// 4. If |new_co2 - avg| > MaxCo2Delta → Reject as physically implausible
        /// 5. Otherwise → Accept
        ///
        /// # Why This Matters for Grant
        /// This demonstrates that TrionChain doesn't just store data - it actively
        /// enforces physical laws/constraints at the consensus layer. This makes
        /// the chain suitable for critical infrastructure monitoring where data
        /// quality and physical consistency are paramount.
        fn validate_co2_against_neighbors(cell_id: u32, new_co2: u32) -> DispatchResult {
            // Get neighbors for this cell
            let neighbors = match Neighbors::<T>::get(cell_id) {
                Some(n) if !n.is_empty() => n,
                _ => return Ok(()), // No neighbors configured → allow (isolated cell)
            };

            // Collect CO2 levels from neighbors that have data
            let mut neighbor_co2_levels = Vec::new();
            for neighbor_id in neighbors.iter() {
                if let Some(neighbor_data) = CellStates::<T>::get(neighbor_id) {
                    neighbor_co2_levels.push(neighbor_data.co2_level);
                }
            }

            // If no neighbor data exists yet → allow (bootstrap phase)
            if neighbor_co2_levels.is_empty() {
                return Ok(());
            }

            // Calculate average CO2 from neighbors
            let sum: u64 = neighbor_co2_levels.iter()
                .map(|&x| x as u64)
                .sum();
            let avg_co2 = (sum / neighbor_co2_levels.len() as u64) as u32;

            // Calculate absolute delta
            let delta = if new_co2 > avg_co2 {
                new_co2 - avg_co2
            } else {
                avg_co2 - new_co2
            };

            // Enforce spatial consistency threshold
            ensure!(
                delta <= T::MaxCo2Delta::get(),
                Error::<T>::Co2DeltaTooHigh
            );

            Ok(())
        }
    }
}