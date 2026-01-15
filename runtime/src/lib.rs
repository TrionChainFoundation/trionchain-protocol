#![cfg_attr(not(feature = "std"), no_std)]
#![recursion_limit = "256"]

#[cfg(feature = "std")]
pub mod wasm_binary;
#[cfg(feature = "std")]
pub use wasm_binary::WASM_BINARY;

use frame_support::{
	construct_runtime, derive_impl, parameter_types,
	traits::{ConstU32, ConstU64, ConstU8},
	weights::IdentityFee,
};

use sp_api::{decl_runtime_apis, impl_runtime_apis};
use sp_consensus_aura::sr25519::AuthorityId as AuraId;
use sp_core::OpaqueMetadata;
use sp_runtime::{
	create_runtime_str,
	generic,
	traits::{Block as BlockT, IdentifyAccount, Verify},
	transaction_validity::{TransactionSource, TransactionValidity},
	ApplyExtrinsicResult,
	MultiSignature,
	RuntimeString,
};
use sp_std::prelude::*;
use sp_version::RuntimeVersion;

// ------------------------------------------------------------

decl_runtime_apis! {
	pub trait GenesisBuilder {
		fn create_default_config() -> sp_std::vec::Vec<u8>;
		fn build_config(json: sp_std::vec::Vec<u8>) -> Result<(), RuntimeString>;
	}
}

	

// -------------------- TIPOS BÁSICOS --------------------

pub type Signature = MultiSignature;
pub type AccountId = <<Signature as Verify>::Signer as IdentifyAccount>::AccountId;
pub type Address = sp_runtime::MultiAddress<AccountId, ()>;

pub type Balance = u128;
pub type Nonce = u32;

pub type BlockNumber = u32;
pub type Index = u32;

pub type Hash = sp_core::H256;
pub type Header = generic::Header<BlockNumber, sp_runtime::traits::BlakeTwo256>;

pub type SignedExtra = (
	frame_system::CheckNonZeroSender<Runtime>,
	frame_system::CheckSpecVersion<Runtime>,
	frame_system::CheckTxVersion<Runtime>,
	frame_system::CheckGenesis<Runtime>,
	frame_system::CheckMortality<Runtime>,
	frame_system::CheckNonce<Runtime>,
	frame_system::CheckWeight<Runtime>,
	pallet_transaction_payment::ChargeTransactionPayment<Runtime>,
);

pub type UncheckedExtrinsic =
	generic::UncheckedExtrinsic<Address, RuntimeCall, Signature, SignedExtra>;

pub type Block = generic::Block<Header, UncheckedExtrinsic>;

pub mod opaque {
	use super::*;
	use sp_runtime::impl_opaque_keys;

	pub type Aura = sp_consensus_aura::sr25519::AuthorityId;
	pub type Grandpa = sp_consensus_grandpa::AuthorityId;

	impl_opaque_keys! {
		pub struct SessionKeys {
			pub aura: Aura,
			pub grandpa: Grandpa,
		}
	}

	pub type UncheckedExtrinsic = sp_runtime::OpaqueExtrinsic;
	pub type Block = sp_runtime::generic::Block<Header, UncheckedExtrinsic>;
	pub type BlockId = sp_runtime::generic::BlockId<Block>;
}

// -------------------- VERSION --------------------

#[sp_version::runtime_version]
pub const VERSION: RuntimeVersion = RuntimeVersion {
	spec_name: create_runtime_str!("trionchain-runtime"),
	impl_name: create_runtime_str!("trionchain-runtime"),
	authoring_version: 1,
	spec_version: 100,
	impl_version: 1,

	// ✅ lo genera impl_runtime_apis!
	apis: RUNTIME_API_VERSIONS,

	transaction_version: 1,
	state_version: 1,
};

#[cfg(feature = "std")]
pub fn native_version() -> sp_version::NativeVersion {
	sp_version::NativeVersion {
		runtime_version: VERSION,
		can_author_with: Default::default(),
	}
}

parameter_types! {
	pub const BlockHashCount: BlockNumber = 2400;
	pub const Version: RuntimeVersion = VERSION;
	pub const SS58Prefix: u16 = 42;
}

// -------------------- SYSTEM --------------------

#[derive_impl(frame_system::config_preludes::SolochainDefaultConfig as frame_system::DefaultConfig)]
impl frame_system::Config for Runtime {
	type RuntimeEvent = RuntimeEvent;
	type RuntimeCall = RuntimeCall;

	type AccountId = AccountId;
	type Lookup = sp_runtime::traits::AccountIdLookup<AccountId, ()>;
	type Nonce = Index;
	type Hash = Hash;

	type Block = Block;

	type BlockHashCount = BlockHashCount;
	type Version = Version;
	type SS58Prefix = SS58Prefix;

	type PalletInfo = PalletInfo;

	type AccountData = pallet_balances::AccountData<Balance>;
	type OnNewAccount = ();
	type OnKilledAccount = ();
	type DbWeight = ();
	type SystemWeightInfo = ();
	type BlockWeights = ();
	type BlockLength = ();

	type RuntimeTask = ();

	type MaxConsumers = ConstU32<16>;
}

// -------------------- TIMESTAMP --------------------

parameter_types! {
	pub const MinimumPeriod: u64 = 1000;
}

impl pallet_timestamp::Config for Runtime {
	type Moment = u64;
	type OnTimestampSet = Aura;
	type MinimumPeriod = MinimumPeriod;
	type WeightInfo = ();
}

// -------------------- AURA --------------------

impl pallet_aura::Config for Runtime {
	type AuthorityId = AuraId;
	type DisabledValidators = ();
	type MaxAuthorities = ConstU32<32>;
	type AllowMultipleBlocksPerSlot = frame_support::traits::ConstBool<false>;
}

// -------------------- GRANDPA --------------------

impl pallet_grandpa::Config for Runtime {
	type RuntimeEvent = RuntimeEvent;
	type MaxNominators = ConstU32<0>;
	type WeightInfo = ();
	type MaxAuthorities = ConstU32<32>;
	type MaxSetIdSessionEntries = ConstU64<0>;
	type KeyOwnerProof = sp_core::Void;
	type EquivocationReportSystem = ();
}

// -------------------- BALANCES --------------------

parameter_types! {
	pub const ExistentialDeposit: Balance = 500;
	pub const MaxLocks: u32 = 50;
}

impl pallet_balances::Config for Runtime {
	type RuntimeEvent = RuntimeEvent;
	type Balance = Balance;

	type DustRemoval = ();
	type ExistentialDeposit = ExistentialDeposit;
	type AccountStore = System;
	type WeightInfo = ();

	type MaxLocks = MaxLocks;
	type MaxReserves = ();
	type ReserveIdentifier = [u8; 8];

	type RuntimeHoldReason = ();
	type RuntimeFreezeReason = ();
	type FreezeIdentifier = ();
	type MaxHolds = ();
	type MaxFreezes = ();
}

// -------------------- TRANSACTION PAYMENT --------------------

impl pallet_transaction_payment::Config for Runtime {
	type RuntimeEvent = RuntimeEvent;

	type OnChargeTransaction =
		pallet_transaction_payment::CurrencyAdapter<Balances, ()>;

	type WeightToFee = IdentityFee<Balance>;
	type LengthToFee = IdentityFee<Balance>;

	type OperationalFeeMultiplier = ConstU8<5>;
	type FeeMultiplierUpdate = ();
}

// -------------------- SUDO --------------------

impl pallet_sudo::Config for Runtime {
	type RuntimeEvent = RuntimeEvent;
	type RuntimeCall = RuntimeCall;
	type WeightInfo = ();
}

// -------------------- CONSTRUCT RUNTIME --------------------

construct_runtime!(
	pub enum Runtime where
		Block = Block,
		NodeBlock = opaque::Block,
		UncheckedExtrinsic = UncheckedExtrinsic,
	{
		System: frame_system,
		Timestamp: pallet_timestamp,
		Aura: pallet_aura,
		Grandpa: pallet_grandpa,
		Balances: pallet_balances,
		TransactionPayment: pallet_transaction_payment,
		Sudo: pallet_sudo,
	}
);

// -------------------- EXECUTIVE --------------------

pub type SignedBlock = generic::SignedBlock<Block>;
pub type SignedPayload = generic::SignedPayload<RuntimeCall, SignedExtra>;

pub type Executive = frame_executive::Executive<
	Runtime,
	Block,
	frame_system::ChainContext<Runtime>,
	Runtime,
	AllPalletsWithSystem,
>;

// -------------------- RUNTIME APIS --------------------

impl_runtime_apis! {
	impl sp_api::Core<Block> for Runtime {
		fn version() -> RuntimeVersion { VERSION }

		fn execute_block(block: Block) {
			Executive::execute_block(block);
		}

		fn initialize_block(header: &<Block as BlockT>::Header) {
			Executive::initialize_block(header)
		}
	}

	impl sp_api::Metadata<Block> for Runtime {
		fn metadata() -> OpaqueMetadata {
			OpaqueMetadata::new(Runtime::metadata().into())
		}

		fn metadata_at_version(version: u32) -> Option<OpaqueMetadata> {
			Runtime::metadata_at_version(version)
		}

		fn metadata_versions() -> sp_std::vec::Vec<u32> {
			Runtime::metadata_versions()
		}
	}

	impl sp_block_builder::BlockBuilder<Block> for Runtime {
		fn apply_extrinsic(extrinsic: <Block as BlockT>::Extrinsic) -> ApplyExtrinsicResult {
			Executive::apply_extrinsic(extrinsic)
		}

		fn finalize_block() -> <Block as BlockT>::Header {
			Executive::finalize_block()
		}

		fn inherent_extrinsics(data: sp_inherents::InherentData) -> Vec<<Block as BlockT>::Extrinsic> {
			data.create_extrinsics()
		}

		fn check_inherents(
			block: Block,
			data: sp_inherents::InherentData,
		) -> sp_inherents::CheckInherentsResult {
			data.check_extrinsics(&block)
		}
	}

	impl sp_transaction_pool::runtime_api::TaggedTransactionQueue<Block> for Runtime {
		fn validate_transaction(
			source: TransactionSource,
			tx: <Block as BlockT>::Extrinsic,
			block_hash: <Block as BlockT>::Hash,
		) -> TransactionValidity {
			Executive::validate_transaction(source, tx, block_hash)
		}
	}

	impl sp_offchain::OffchainWorkerApi<Block> for Runtime {
		fn offchain_worker(header: &<Block as BlockT>::Header) {
			Executive::offchain_worker(header)
		}
	}

	impl sp_consensus_aura::AuraApi<Block, AuraId> for Runtime {
		fn slot_duration() -> sp_consensus_aura::SlotDuration {
			sp_consensus_aura::SlotDuration::from_millis(Aura::slot_duration())
		}

		fn authorities() -> Vec<AuraId> {
			Aura::authorities().into_inner()
		}
	}

	impl sp_consensus_grandpa::GrandpaApi<Block> for Runtime {
		fn grandpa_authorities() -> sp_consensus_grandpa::AuthorityList {
			Grandpa::grandpa_authorities()
		}

		fn current_set_id() -> sp_consensus_grandpa::SetId {
			Grandpa::current_set_id()
		}

		fn submit_report_equivocation_unsigned_extrinsic(
			_equivocation_proof: sp_consensus_grandpa::EquivocationProof<
				<Block as BlockT>::Hash,
				sp_runtime::traits::NumberFor<Block>
			>,
			_key_owner_proof: sp_consensus_grandpa::OpaqueKeyOwnershipProof,
		) -> Option<()> {
			None
		}

		fn generate_key_ownership_proof(
			_set_id: sp_consensus_grandpa::SetId,
			_authority_id: sp_consensus_grandpa::AuthorityId,
		) -> Option<sp_consensus_grandpa::OpaqueKeyOwnershipProof> {
			None
		}
	}

	impl frame_system_rpc_runtime_api::AccountNonceApi<Block, AccountId, Nonce> for Runtime {
		fn account_nonce(account: AccountId) -> Nonce {
			System::account_nonce(account)
		}
	}

	impl pallet_transaction_payment_rpc_runtime_api::TransactionPaymentApi<Block, Balance> for Runtime {
		fn query_info(
			uxt: <Block as BlockT>::Extrinsic,
			len: u32,
		) -> pallet_transaction_payment_rpc_runtime_api::RuntimeDispatchInfo<Balance> {
			TransactionPayment::query_info(uxt, len)
		}

		fn query_fee_details(
			uxt: <Block as BlockT>::Extrinsic,
			len: u32,
		) -> pallet_transaction_payment::FeeDetails<Balance> {
			TransactionPayment::query_fee_details(uxt, len)
		}

		fn query_weight_to_fee(weight: frame_support::weights::Weight) -> Balance {
			TransactionPayment::weight_to_fee(weight)
		}

		fn query_length_to_fee(length: u32) -> Balance {
			TransactionPayment::length_to_fee(length)
		}
	}

	impl sp_session::SessionKeys<Block> for Runtime {
		fn generate_session_keys(seed: Option<Vec<u8>>) -> Vec<u8> {
			opaque::SessionKeys::generate(seed)
		}

		fn decode_session_keys(encoded: Vec<u8>)
			-> Option<Vec<(Vec<u8>, sp_core::crypto::KeyTypeId)>>
		{
			opaque::SessionKeys::decode_into_raw_public_keys(&encoded)
		}
	}

	// ✅ LA CLAVE: el trait necesita <Block>
	impl crate::GenesisBuilder<Block> for Runtime {
		fn create_default_config() -> sp_std::vec::Vec<u8> {
			serde_json::to_vec(&RuntimeGenesisConfig::default())
				.unwrap_or_default()
		}

		fn build_config(_json: sp_std::vec::Vec<u8>) -> Result<(), RuntimeString> {
			Ok(())
		}
	}
}
