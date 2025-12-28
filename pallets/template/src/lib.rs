#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;
#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;
pub mod weights;
pub use weights::*;

#[frame_support::pallet]
pub mod pallet {
	use super::*;
	use frame_support::pallet_prelude::*;
	use frame_system::pallet_prelude::*;

	#[pallet::pallet]
	pub struct Pallet<T>(_);

	/// Configuración del módulo
	#[pallet::config]
	pub trait Config: frame_system::Config {
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
		type WeightInfo: WeightInfo;
	}

	// --- 1. ESTRUCTURAS DE DATOS ---
	// Incluye MaxEncodedLen, Copy, Eq
	#[derive(Encode, Decode, Clone, Copy, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
	pub struct TrionCellData {
		pub stress: u32,      
		pub generation: u32,  
		pub demand: u32,      
		pub soc: u32,         
		pub price: u32,       
	}

	// --- 2. ALMACENAMIENTO (STORAGE) ---
	
	#[pallet::storage]
	#[pallet::getter(fn get_cell_state)]
	pub type CellState<T: Config> = StorageMap<_, Blake2_128Concat, u32, TrionCellData, OptionQuery>;

	#[pallet::storage]
	#[pallet::getter(fn get_authorized_sensor)]
	pub type TrustedSensors<T: Config> = StorageMap<_, Blake2_128Concat, u32, T::AccountId, OptionQuery>;

	// --- 3. EVENTOS ---
	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Datos reportados (Incluye soc/fuel ahora)
		CellUpdateReceived { 
			cell_id: u32, 
			who: T::AccountId, 
			stress: u32,
			generation: u32,
			soc: u32 
		},
		SensorAuthorized { cell_id: u32, operator: T::AccountId },
	}

	// --- 4. ERRORES ---
	#[pallet::error]
	pub enum Error<T> {
		InvalidPhysicalValue, 
		Unauthorized,         
		SensorNotRegistered,  
	}

	// --- 5. FUNCIONES (TRANSACCIONES) ---
	#[pallet::call]
	impl<T: Config> Pallet<T> {
		
		#[pallet::call_index(0)]
		#[pallet::weight(T::WeightInfo::do_something())]
		pub fn register_sensor(origin: OriginFor<T>, cell_id: u32, sensor_account: T::AccountId) -> DispatchResult {
			let _who = ensure_signed(origin)?; 
			<TrustedSensors<T>>::insert(cell_id, &sensor_account);
			Self::deposit_event(Event::SensorAuthorized { cell_id, operator: sensor_account });
			Ok(())
		}

		#[pallet::call_index(1)]
		#[pallet::weight(T::WeightInfo::do_something())]
		pub fn report_state(
			origin: OriginFor<T>, 
			cell_id: u32, 
			stress: u32, 
			generation: u32, 
			demand: u32, 
			soc: u32, 
			price: u32
		) -> DispatchResult {
			let who = ensure_signed(origin)?;

			let authorized_sensor = <TrustedSensors<T>>::get(cell_id).ok_or(Error::<T>::SensorNotRegistered)?;
			ensure!(who == authorized_sensor, Error::<T>::Unauthorized);

			if stress > 1000 || soc > 100 {
				return Err(Error::<T>::InvalidPhysicalValue.into());
			}

			let new_data = TrionCellData { stress, generation, demand, soc, price };
			<CellState<T>>::insert(cell_id, &new_data);

			// Evento actualizado con SOC
			Self::deposit_event(Event::CellUpdateReceived { 
				cell_id, 
				who, 
				stress, 
				generation,
				soc 
			});

			Ok(())
		}
	}
}