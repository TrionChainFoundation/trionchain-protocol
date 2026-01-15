//! Service and ServiceFactory implementation. Specialized wrapper over substrate service.

use futures::FutureExt;
use trionchain_runtime::{self, opaque::Block, RuntimeApi};

use sc_client_api::{Backend, BlockBackend};
use sc_consensus_aura::{ImportQueueParams, SlotProportion, StartAuraParams};
use sc_consensus_grandpa::SharedVoterState;
use sc_service::{error::Error as ServiceError, Configuration, TaskManager};
use sc_telemetry::{Telemetry, TelemetryWorker};
use sc_transaction_pool_api::OffchainTransactionPoolFactory;

use sp_consensus_aura::sr25519::AuthorityPair as AuraPair;

use std::{sync::Arc, time::Duration};

pub(crate) type FullClient = sc_service::TFullClient<
	Block,
	RuntimeApi,
	sc_executor::WasmExecutor<sp_io::SubstrateHostFunctions>,
>;
type FullBackend = sc_service::TFullBackend<Block>;
type FullSelectChain = sc_consensus::LongestChain<FullBackend, Block>;

/// The minimum period of blocks on which justifications will be imported and generated.
const GRANDPA_JUSTIFICATION_PERIOD: u32 = 512;

pub type Service = sc_service::PartialComponents<
	FullClient,
	FullBackend,
	FullSelectChain,
	sc_consensus::DefaultImportQueue<Block>,
	sc_transaction_pool::FullPool<Block, FullClient>,
	(
		Option<sc_consensus_grandpa::GrandpaBlockImport<FullBackend, Block, FullClient, FullSelectChain>>,
		Option<sc_consensus_grandpa::LinkHalf<Block, FullClient, FullSelectChain>>,
		Option<Telemetry>,
	),
>;

pub fn new_partial(config: &Configuration) -> Result<Service, ServiceError> {
	let telemetry = config
		.telemetry_endpoints
		.clone()
		.filter(|x| !x.is_empty())
		.map(|endpoints| -> Result<_, sc_telemetry::Error> {
			let worker = TelemetryWorker::new(16)?;
			let telemetry = worker.handle().new_telemetry(endpoints);
			Ok((worker, telemetry))
		})
		.transpose()?;

	let executor = sc_service::new_wasm_executor::<sp_io::SubstrateHostFunctions>(config);
	let (client, backend, keystore_container, task_manager) =
		sc_service::new_full_parts::<Block, RuntimeApi, _>(
			config,
			telemetry.as_ref().map(|(_, telemetry)| telemetry.handle()),
			executor,
		)?;
	let client = Arc::new(client);

	let telemetry = telemetry.map(|(worker, telemetry)| {
		task_manager.spawn_handle().spawn("telemetry", None, worker.run());
		telemetry
	});

	let select_chain = sc_consensus::LongestChain::new(backend.clone());

	let transaction_pool = sc_transaction_pool::BasicPool::new_full(
		config.transaction_pool.clone(),
		config.role.is_authority().into(),
		config.prometheus_registry(),
		task_manager.spawn_essential_handle(),
		client.clone(),
	);

	// ✅ Detect dev chain: in dev we DO NOT init GRANDPA (it panics in aux_schema load_persistent in your SDK).
	let is_dev_chain = matches!(config.chain_spec.chain_type(), sc_service::ChainType::Development);

	let (maybe_grandpa_block_import, maybe_grandpa_link) = if is_dev_chain {
		(None, None)
	} else {
		let (grandpa_block_import, grandpa_link) = sc_consensus_grandpa::block_import(
			client.clone(),
			GRANDPA_JUSTIFICATION_PERIOD,
			&client,
			select_chain.clone(),
			telemetry.as_ref().map(|x| x.handle()),
		)?;
		(Some(grandpa_block_import), Some(grandpa_link))
	};

	let slot_duration = sc_consensus_aura::slot_duration(&*client)?;

	// ✅ Aura import queue:
	// - Normal chains: use grandpa_block_import + justification_import
	// - Dev chain: use client, no justification_import
	let import_queue = if let Some(ref grandpa_block_import) = maybe_grandpa_block_import {
		sc_consensus_aura::import_queue::<AuraPair, _, _, _, _, _>(ImportQueueParams {
			block_import: grandpa_block_import.clone(),
			justification_import: Some(Box::new(grandpa_block_import.clone())),
			client: client.clone(),
			create_inherent_data_providers: move |_, ()| async move {
				let timestamp = sp_timestamp::InherentDataProvider::from_system_time();
				let slot =
					sp_consensus_aura::inherents::InherentDataProvider::from_timestamp_and_slot_duration(
						*timestamp,
						slot_duration,
					);
				Ok((slot, timestamp))
			},
			spawner: &task_manager.spawn_essential_handle(),
			registry: config.prometheus_registry(),
			check_for_equivocation: Default::default(),
			telemetry: telemetry.as_ref().map(|x| x.handle()),
			compatibility_mode: Default::default(),
		})?
	} else {
		sc_consensus_aura::import_queue::<AuraPair, _, _, _, _, _>(ImportQueueParams {
			block_import: client.clone(),
			justification_import: None,
			client: client.clone(),
			create_inherent_data_providers: move |_, ()| async move {
				let timestamp = sp_timestamp::InherentDataProvider::from_system_time();
				let slot =
					sp_consensus_aura::inherents::InherentDataProvider::from_timestamp_and_slot_duration(
						*timestamp,
						slot_duration,
					);
				Ok((slot, timestamp))
			},
			spawner: &task_manager.spawn_essential_handle(),
			registry: config.prometheus_registry(),
			check_for_equivocation: Default::default(),
			telemetry: telemetry.as_ref().map(|x| x.handle()),
			compatibility_mode: Default::default(),
		})?
	};

	Ok(sc_service::PartialComponents {
		client,
		backend,
		task_manager,
		import_queue,
		keystore_container,
		select_chain,
		transaction_pool,
		other: (maybe_grandpa_block_import, maybe_grandpa_link, telemetry),
	})
}

/// Builds a new service for a full client.
pub fn new_full(config: Configuration) -> Result<TaskManager, ServiceError> {
	let sc_service::PartialComponents {
		client,
		backend,
		mut task_manager,
		import_queue,
		keystore_container,
		select_chain,
		transaction_pool,
		other: (maybe_block_import, maybe_grandpa_link, mut telemetry),
	} = new_partial(&config)?;

	let mut net_config = sc_network::config::FullNetworkConfiguration::new(&config.network);

	// ✅ Dev chain: disable GRANDPA always (avoid panic in your SDK)
	let is_dev_chain = matches!(config.chain_spec.chain_type(), sc_service::ChainType::Development);
	let enable_grandpa = !config.disable_grandpa && !is_dev_chain;

	// Add GRANDPA protocol only if enabled
	let (grandpa_protocol_name, grandpa_notification_service, grandpa_link) = if enable_grandpa {
		let grandpa_link = maybe_grandpa_link.expect("GRANDPA enabled => link exists; qed");

		let grandpa_protocol_name = sc_consensus_grandpa::protocol_standard_name(
			&client.block_hash(0).ok().flatten().expect("Genesis exists; qed"),
			&config.chain_spec,
		);

		let (grandpa_protocol_config, grandpa_notification_service) =
			sc_consensus_grandpa::grandpa_peers_set_config(grandpa_protocol_name.clone());

		net_config.add_notification_protocol(grandpa_protocol_config);

		(Some(grandpa_protocol_name), Some(grandpa_notification_service), Some(grandpa_link))
	} else {
		(None, None, None)
	};

	let (network, system_rpc_tx, tx_handler_controller, network_starter, sync_service) =
		sc_service::build_network(sc_service::BuildNetworkParams {
			config: &config,
			net_config,
			client: client.clone(),
			transaction_pool: transaction_pool.clone(),
			spawn_handle: task_manager.spawn_handle(),
			import_queue,
			block_announce_validator_builder: None,
			// ✅ No warp sync (your SDK doesn’t expose sc_network::warp and dev doesn’t need it)
			warp_sync_params: None,
			block_relay: None,
		})?;

	if config.offchain_worker.enabled {
		task_manager.spawn_handle().spawn(
			"offchain-workers-runner",
			"offchain-worker",
			sc_offchain::OffchainWorkers::new(sc_offchain::OffchainWorkerOptions {
				runtime_api_provider: client.clone(),
				is_validator: config.role.is_authority(),
				keystore: Some(keystore_container.keystore()),
				offchain_db: backend.offchain_storage(),
				transaction_pool: Some(OffchainTransactionPoolFactory::new(transaction_pool.clone())),
				network_provider: network.clone(),
				enable_http_requests: true,
				custom_extensions: |_| vec![],
			})
			.run(client.clone(), task_manager.spawn_handle())
			.boxed(),
		);
	}

	let role = config.role.clone();
	let force_authoring = config.force_authoring;
	let backoff_authoring_blocks: Option<()> = None;
	let name = config.network.node_name.clone();
	let prometheus_registry = config.prometheus_registry().cloned();

	let rpc_extensions_builder = {
		let client = client.clone();
		let pool = transaction_pool.clone();

		Box::new(move |deny_unsafe, _| {
			let deps =
				crate::rpc::FullDeps { client: client.clone(), pool: pool.clone(), deny_unsafe };
			crate::rpc::create_full(deps).map_err(Into::into)
		})
	};

	let _rpc_handlers = sc_service::spawn_tasks(sc_service::SpawnTasksParams {
		network: network.clone(),
		client: client.clone(),
		keystore: keystore_container.keystore(),
		task_manager: &mut task_manager,
		transaction_pool: transaction_pool.clone(),
		rpc_builder: rpc_extensions_builder,
		backend,
		system_rpc_tx,
		tx_handler_controller,
		sync_service: sync_service.clone(),
		config,
		telemetry: telemetry.as_mut(),
	})?;

	// ✅ Aura authoring:
	// - Normal chains: use grandpa block_import
	// - Dev chain: use client as block_import
	if role.is_authority() {
		let proposer_factory = sc_basic_authorship::ProposerFactory::new(
			task_manager.spawn_handle(),
			client.clone(),
			transaction_pool.clone(),
			prometheus_registry.as_ref(),
			telemetry.as_ref().map(|x| x.handle()),
		);

		let slot_duration = sc_consensus_aura::slot_duration(&*client)?;

		if let Some(block_import) = maybe_block_import.clone() {
			// NORMAL (with GRANDPA block_import)
			let aura = sc_consensus_aura::start_aura::<AuraPair, _, _, _, _, _, _, _, _, _, _>(
				StartAuraParams {
					slot_duration,
					client: client.clone(),
					select_chain: select_chain.clone(),
					block_import,
					proposer_factory,
					create_inherent_data_providers: move |_, ()| async move {
						let timestamp = sp_timestamp::InherentDataProvider::from_system_time();
						let slot =
							sp_consensus_aura::inherents::InherentDataProvider::from_timestamp_and_slot_duration(
								*timestamp,
								slot_duration,
							);
						Ok((slot, timestamp))
					},
					force_authoring,
					backoff_authoring_blocks,
					keystore: keystore_container.keystore(),
					sync_oracle: sync_service.clone(),
					justification_sync_link: sync_service.clone(),
					block_proposal_slot_portion: SlotProportion::new(2f32 / 3f32),
					max_block_proposal_slot_portion: None,
					telemetry: telemetry.as_ref().map(|x| x.handle()),
					compatibility_mode: Default::default(),
				},
			)?;

			task_manager
				.spawn_essential_handle()
				.spawn_blocking("aura", Some("block-authoring"), aura);
		} else {
			// DEV (no GRANDPA)
			let aura = sc_consensus_aura::start_aura::<AuraPair, _, _, _, _, _, _, _, _, _, _>(
				StartAuraParams {
					slot_duration,
					client: client.clone(),
					select_chain: select_chain.clone(),
					block_import: client.clone(),
					proposer_factory,
					create_inherent_data_providers: move |_, ()| async move {
						let timestamp = sp_timestamp::InherentDataProvider::from_system_time();
						let slot =
							sp_consensus_aura::inherents::InherentDataProvider::from_timestamp_and_slot_duration(
								*timestamp,
								slot_duration,
							);
						Ok((slot, timestamp))
					},
					force_authoring,
					backoff_authoring_blocks,
					keystore: keystore_container.keystore(),
					sync_oracle: sync_service.clone(),
					justification_sync_link: sync_service.clone(),
					block_proposal_slot_portion: SlotProportion::new(2f32 / 3f32),
					max_block_proposal_slot_portion: None,
					telemetry: telemetry.as_ref().map(|x| x.handle()),
					compatibility_mode: Default::default(),
				},
			)?;

			task_manager
				.spawn_essential_handle()
				.spawn_blocking("aura", Some("block-authoring"), aura);
		}
	}

	// ✅ Start GRANDPA voter only if enabled (not dev)
	if enable_grandpa {
		let grandpa_link = grandpa_link.expect("enable_grandpa => link exists; qed");
		let grandpa_protocol_name = grandpa_protocol_name.expect("enable_grandpa => set; qed");
		let grandpa_notification_service =
			grandpa_notification_service.expect("enable_grandpa => set; qed");

		let keystore = if role.is_authority() { Some(keystore_container.keystore()) } else { None };

		let grandpa_config = sc_consensus_grandpa::Config {
			gossip_duration: Duration::from_millis(333),
			justification_generation_period: GRANDPA_JUSTIFICATION_PERIOD,
			name: Some(name),
			observer_enabled: false,
			keystore,
			local_role: role,
			telemetry: telemetry.as_ref().map(|x| x.handle()),
			protocol_name: grandpa_protocol_name,
		};

		let grandpa_params = sc_consensus_grandpa::GrandpaParams {
			config: grandpa_config,
			link: grandpa_link,
			network,
			sync: Arc::new(sync_service),
			notification_service: grandpa_notification_service,
			voting_rule: sc_consensus_grandpa::VotingRulesBuilder::default().build(),
			prometheus_registry,
			shared_voter_state: SharedVoterState::empty(),
			telemetry: telemetry.as_ref().map(|x| x.handle()),
			offchain_tx_pool_factory: OffchainTransactionPoolFactory::new(transaction_pool),
		};

		task_manager.spawn_essential_handle().spawn_blocking(
			"grandpa-voter",
			None,
			sc_consensus_grandpa::run_grandpa_voter(grandpa_params)?,
		);
	}

	network_starter.start_network();
	Ok(task_manager)
}
