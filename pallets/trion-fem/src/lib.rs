#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::{
        pallet_prelude::*,
        BoundedVec,
    };
    use frame_system::pallet_prelude::*;
    use sp_std::vec::Vec;

    pub type MaxPhysicsLen = ConstU32<256>;

    #[derive(
        Encode, Decode, Clone, PartialEq, Eq,
        RuntimeDebug, TypeInfo, MaxEncodedLen
    )]
    pub struct TrionCellData {
        pub step: u32,
        pub physics: BoundedVec<u32, MaxPhysicsLen>,
        pub updated_at_block: u64,
    }

    #[pallet::config]
    pub trait Config: frame_system::Config {}

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::storage]
    #[pallet::getter(fn cell_state)]
    pub type CellStates<T: Config> =
        StorageMap<_, Blake2_128Concat, u32, TrionCellData, OptionQuery>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        CellUpdated { cell_id: u32, step: u32, physics_len: u32 },
    }

    #[pallet::error]
    pub enum Error<T> {
        PhysicsTooLong,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        #[pallet::call_index(0)]
        #[pallet::weight(T::DbWeight::get().reads_writes(1, 1))]
        pub fn update_cell(
            origin: OriginFor<T>,
            cell_id: u32,
            step: u32,
            physics: Vec<u32>,
        ) -> DispatchResult {
            let _ = ensure_signed(origin)?;

            let bounded: BoundedVec<u32, MaxPhysicsLen> =
                physics.try_into().map_err(|_| Error::<T>::PhysicsTooLong)?;

            use sp_runtime::traits::SaturatedConversion;
            let now: u64 =
                frame_system::Pallet::<T>::block_number().saturated_into();

            let data = TrionCellData {
                step,
                physics: bounded,
                updated_at_block: now,
            };

            CellStates::<T>::insert(cell_id, data);

            let physics_len = CellStates::<T>::get(cell_id)
                .map(|d| d.physics.len() as u32)
                .unwrap_or(0);

            Self::deposit_event(Event::CellUpdated {
                cell_id,
                step,
                physics_len,
            });

            Ok(())
        }
    }
}
