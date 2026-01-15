// runtime/src/wasm_binary.rs
#![cfg(feature = "std")]

// Este archivo lo genera substrate-wasm-builder dentro de OUT_DIR.
// Define WASM_BINARY (y a veces WASM_BINARY_BLOATY) según versión.
include!(concat!(env!("OUT_DIR"), "/wasm_binary.rs"));
