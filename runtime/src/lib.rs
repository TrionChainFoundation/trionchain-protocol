#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::{
    construct_runtime, parameter_types,
    traits::{ConstU16, ConstU32, Everything},
};
use frame_system::limits::{BlockLength, BlockWeights};
use sp_core::{H256, OpaqueMetadata};
use sp_runtime::{
    generic,
    traits::{BlakeTwo256, IdentifyAccount, Verify},
    transaction_validity::{TransactionSource, TransactionValidity},
    ApplyExtrinsicResult, ExtrinsicInclusionMode, MultiSignature, Perbill, Weight,
};
use sp_std::{borrow::Cow, vec::Vec};
use sp_version::RuntimeVersion;

// ---------------- Types básicos ----------------
pub type BlockNumber = u32;
pub type Balance = u128;
pub type Nonce = u32;
pub type Hash = H256;

pub type Signature = MultiSignature;
pub type AccountId = <<Signature as Verify>::Signer as IdentifyAccount>::AccountId;

pub type Header = generic::Header<BlockNumber, BlakeTwo256>;

pub type SignedExtra = (
    frame_system::CheckNonZeroSender<Runtime>,
    frame_system::CheckSpecVersion<Runtime>,
    frame_system::CheckTxVersion<Runtime>,
    frame_system::CheckGenesis<Runtime>,
    frame_system::CheckEra<Runtime>,
    frame_system::CheckNonce<Runtime>,
    frame_system::CheckWeight<Runtime>,
);

pub type UncheckedExtrinsic =
    generic::UncheckedExtrinsic<AccountId, RuntimeCall, Signature, SignedExtra>;
pub type Block = generic::Block<Header, UncheckedExtrinsic>;

// ---------------- RuntimeVersion ----------------
#[sp_version::runtime_version]
pub const VERSION: RuntimeVersion = RuntimeVersion {
    spec_name: Cow::Borrowed("trionchain"),
    impl_name: Cow::Borrowed("trionchain"),
    authoring_version: 1,
    spec_version: 1,
    impl_version: 1,
    apis: RUNTIME_API_VERSIONS,
    transaction_version: 1,
    system_version: 1,
};

// ---------------- Parámetros ----------------
parameter_types! {
    pub const BlockHashCount: u32 = 2400;
    pub const SS58Prefix: u16 = 42;

    pub RuntimeBlockLength: BlockLength =
        BlockLength::max_with_normal_ratio(5 * 1024 * 1024, Perbill::from_percent(75));

    pub RuntimeBlockWeights: BlockWeights =
        BlockWeights::simple_max(Weight::from_parts(2_000_000_000, 0));

    pub const Version: RuntimeVersion = VERSION;
    pub const ExistentialDeposit: Balance = 1;
}

// ---------------- frame_system ----------------
impl frame_system::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type BaseCallFilter = Everything;
    type Block = Block;
    type RuntimeCall = RuntimeCall;
    type RuntimeOrigin = RuntimeOrigin;

    type AccountId = AccountId;
    type Lookup = sp_runtime::traits::IdentityLookup<AccountId>;
    type Nonce = Nonce;
    type Hash = Hash;
    type Hashing = BlakeTwo256;

    type BlockHashCount = BlockHashCount;
    type BlockWeights = RuntimeBlockWeights;
    type BlockLength = RuntimeBlockLength;

    type DbWeight = ();
    type Version = Version;
    type PalletInfo = PalletInfo;
    type AccountData = pallet_balances::AccountData<Balance>;
    type OnNewAccount = ();
    type OnKilledAccount = ();
    type SystemWeightInfo = ();
    type SS58Prefix = ConstU16<42>;
    type OnSetCode = ();

    type MaxConsumers = ConstU32<16>;

    type RuntimeTask = ();
    type ExtensionsWeightInfo = ();
    type SingleBlockMigrations = ();
    type MultiBlockMigrator = ();
    type PreInherents = ();
    type PostInherents = ();
    type PostTransactions = ();
}

// ---------------- Balances ----------------
impl pallet_balances::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type Balance = Balance;
    type DustRemoval = ();
    type ExistentialDeposit = ExistentialDeposit;
    type AccountStore = System;
    type WeightInfo = ();
    type MaxLocks = ConstU32<50>;
    type MaxReserves = ConstU32<50>;
    type ReserveIdentifier = [u8; 8];

    type RuntimeHoldReason = RuntimeHoldReason;
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type FreezeIdentifier = [u8; 8];
    type MaxFreezes = ConstU32<0>;
    type DoneSlashHandler = ();
}

// ---------------- Sudo ----------------
impl pallet_sudo::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type RuntimeCall = RuntimeCall;
    type WeightInfo = ();
}

// ---------------- Trion FEM ----------------
impl pallet_trion_fem::Config for Runtime {}

// ---------------- Runtime ----------------
construct_runtime!(
    pub enum Runtime {
        System: frame_system,
        Balances: pallet_balances,
        Sudo: pallet_sudo,
        TrionFem: pallet_trion_fem,
    }
);

// ---------------- Executive ----------------
pub type Executive = frame_executive::Executive<
    Runtime,
    Block,
    frame_system::ChainContext<Runtime>,
    Runtime,
    AllPalletsWithSystem,
>;

// ---------------- Runtime APIs ----------------
sp_api::impl_runtime_apis! {
    impl sp_api::Core<Block> for Runtime {
        fn version() -> RuntimeVersion { VERSION }

        fn execute_block(block: <Block as sp_runtime::traits::Block>::LazyBlock) {
            Executive::execute_block(block)
        }

        fn initialize_block(
            header: &<Block as sp_runtime::traits::Block>::Header
        ) -> ExtrinsicInclusionMode {
            Executive::initialize_block(header)
        }
    }

    impl sp_api::Metadata<Block> for Runtime {
        fn metadata() -> OpaqueMetadata {
        // En tu caso, esto sigue siendo correcto:
           OpaqueMetadata::new(Runtime::metadata().into())
        }

        fn metadata_at_version(version: u32) -> Option<OpaqueMetadata> {
        // ✅ YA devuelve OpaqueMetadata, no lo vuelvas a envolver
           Runtime::metadata_at_version(version)
        }

    fn metadata_versions() -> Vec<u32> {
        Runtime::metadata_versions()
        }  
    }


    impl sp_block_builder::BlockBuilder<Block> for Runtime {
        fn apply_extrinsic(
            extrinsic: <Block as sp_runtime::traits::Block>::Extrinsic
        ) -> ApplyExtrinsicResult {
            Executive::apply_extrinsic(extrinsic)
        }

        fn finalize_block() -> <Block as sp_runtime::traits::Block>::Header {
            Executive::finalize_block()
        }

        fn inherent_extrinsics(
            data: sp_inherents::InherentData
        ) -> Vec<<Block as sp_runtime::traits::Block>::Extrinsic> {
            data.create_extrinsics()
        }

        fn check_inherents(
            block: <Block as sp_runtime::traits::Block>::LazyBlock,
            data: sp_inherents::InherentData
        ) -> sp_inherents::CheckInherentsResult {
            data.check_extrinsics(&block)
        }
    }

    impl sp_transaction_pool::runtime_api::TaggedTransactionQueue<Block> for Runtime {
        fn validate_transaction(
            source: TransactionSource,
            tx: <Block as sp_runtime::traits::Block>::Extrinsic,
            block_hash: <Block as sp_runtime::traits::Block>::Hash,
        ) -> TransactionValidity {
            Executive::validate_transaction(source, tx, block_hash)
        }
    }
}
