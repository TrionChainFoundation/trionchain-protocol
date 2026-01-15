// runtime/build.rs
fn main() {
	// Esto hace que cargo re-ejecute el build script si cambias el runtime.
	println!("cargo:rerun-if-changed=src/lib.rs");
	println!("cargo:rerun-if-changed=build.rs");

	// Genera el wasm del runtime y crea el archivo OUT_DIR/wasm_binary.rs
	substrate_wasm_builder::WasmBuilder::new()
		.with_current_project()
		.export_heap_base()
		.import_memory()
		.build();
}
