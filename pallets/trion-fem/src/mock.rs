use crate as pallet_trion_fem;
use frame_support::{derive_impl, parameter_types};
use sp_runtime::BuildStorage;

type Block = frame_system::mocking::MockBlock<Test>;

// Construct mock runtime
frame_support::construct_runtime!(
    pub enum Test {
        System: frame_system,
        TrionFem: pallet_trion_fem,
    }
);

// Configure mock System pallet
#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::Config for Test {
    type Block = Block;
}

// Configure constants for TrionFem pallet
parameter_types! {
    /// Maximum 8 neighbors per cell (octagonal/cubic mesh)
    pub const MaxNeighbors: u32 = 8;
    /// Maximum CO2 deviation: 100 ppm
    /// This is reasonable for adjacent environmental sensors
    pub const MaxCo2Delta: u32 = 100;
}

// Configure TrionFem pallet for testing
impl pallet_trion_fem::Config for Test {
    type WeightInfo = ();
    type MaxNeighbors = MaxNeighbors;
    type MaxCo2Delta = MaxCo2Delta;
}

/// Build test externalities
pub fn new_test_ext() -> sp_io::TestExternalities {
    let t = frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap();
    let mut ext = sp_io::TestExternalities::new(t);
    ext.execute_with(|| System::set_block_number(1));
    ext
}