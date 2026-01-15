#![cfg(feature = "std")]

use crate::{AccountId, BalancesConfig, RuntimeGenesisConfig, SudoConfig};
use serde_json::Value;
use sp_consensus_aura::sr25519::AuthorityId as AuraId;
use sp_consensus_grandpa::AuthorityId as GrandpaId;
// use sp_genesis_builder::{PresetId, DEV_RUNTIME_PRESET, LOCAL_TESTNET_RUNTIME_PRESET};
use sp_keyring::{Ed25519Keyring, Sr25519Keyring};

/// Build a JSON patch (as `serde_json::Value`) for the runtime genesis config.
///
/// Nota:
/// - Esto compila SOLO en `std` (chain-spec / presets), no entra al runtime wasm.
/// - No usa `alloc`, ni macros especiales.
fn testnet_genesis(
    initial_authorities: Vec<(AuraId, GrandpaId)>,
    endowed_accounts: Vec<AccountId>,
    root: AccountId,
) -> Value {
    // Construimos el RuntimeGenesisConfig (tipado) y luego lo serializamos a JSON.
    let genesis = RuntimeGenesisConfig {
        balances: BalancesConfig {
            balances: endowed_accounts
                .into_iter()
                .map(|k| (k, 1u128 << 60))
                .collect(),
        },
        aura: pallet_aura::GenesisConfig {
            authorities: initial_authorities.iter().map(|x| x.0.clone()).collect(),
        },
        grandpa: pallet_grandpa::GenesisConfig {
            authorities: initial_authorities
                .iter()
                .map(|x| (x.1.clone(), 1))
                .collect(),

            _config: Default::default(),
        },
        sudo: SudoConfig { key: Some(root) },

        // El resto de pallets quedan por defecto.
        ..Default::default()
    };

    serde_json::to_value(genesis).expect("RuntimeGenesisConfig must be serializable to JSON")
}

pub fn development_config_genesis() -> Value {
    testnet_genesis(
        vec![(
            Sr25519Keyring::Alice.public().into(),
            Ed25519Keyring::Alice.public().into(),
        )],
        vec![
            Sr25519Keyring::Alice.to_account_id(),
            Sr25519Keyring::Bob.to_account_id(),
            Sr25519Keyring::AliceStash.to_account_id(),
            Sr25519Keyring::BobStash.to_account_id(),
        ],
        Sr25519Keyring::Alice.to_account_id(),
    )
}

pub fn local_config_genesis() -> Value {
    testnet_genesis(
        vec![
            (
                Sr25519Keyring::Alice.public().into(),
                Ed25519Keyring::Alice.public().into(),
            ),
            (
                Sr25519Keyring::Bob.public().into(),
                Ed25519Keyring::Bob.public().into(),
            ),
        ],
        Sr25519Keyring::iter()
            .filter(|v| v != &Sr25519Keyring::One && v != &Sr25519Keyring::Two)
            .map(|v| v.to_account_id())
            .collect(),
        Sr25519Keyring::Alice.to_account_id(),
    )
}
// Returns JSON bytes for the requested preset id (DEV / LOCAL).
/* DISABLED: genesis presets (sp_genesis_builder API mismatch)
pub fn get_preset(id: &PresetId) -> Option<Vec<u8>> {
    let patch = match id.as_ref() {
        DEV_RUNTIME_PRESET => development_config_genesis(),
        LOCAL_TESTNET_RUNTIME_PRESET => local_config_genesis(),
        _ => return None,
    };

    Some(
        serde_json::to_vec(&patch).expect("json serialization should work"),
    )
}

*/
// Supported preset names.
/* DISABLED: genesis presets (sp_genesis_builder API mismatch)
pub fn preset_names() -> Vec<PresetId> {
    vec![
        PresetId::from(DEV_RUNTIME_PRESET),
        PresetId::from(LOCAL_TESTNET_RUNTIME_PRESET),
    ]
}
*/
