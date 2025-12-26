from substrateinterface import SubstrateInterface, Keypair
import time
import random

# 1. CONNECTION: Establish connection to the local TrionChain node
print("🔌 Connecting to TrionChain Local Node...")

try:
    # We connect to the WebSocket port (default: 9944)
    substrate = SubstrateInterface(
        url="ws://127.0.0.1:9944",
        type_registry_preset='substrate-node-template'
    )
except ConnectionRefusedError:
    print("❌ ERROR: Connection refused. Please ensure your Substrate node is running ('./target/release/solochain-template-node --dev').")
    exit()

# 2. IDENTITY: Load 'Alice' account (Default Superuser for DevNets)
# In production, this would be a private key securely loaded from env variables.
keypair = Keypair.create_from_uri('//Alice')

current_block = substrate.get_block_header()['header']['number']
print(f"✅ Successfully connected. Current Block Height: #{current_block}")
print("-" * 60)

# 3. DATA LOOP: Simulating an IoT Sensor reporting to the Blockchain
cell_id = 1

try:
    while True:
        # A) Generate simulated physical data (Stress between 100 and 900)
        # Note: If stress > 1000, the blockchain logic will reject it (as per your Rust code).
        physical_stress = random.randint(100, 900)
        
        print(f"📡 SENSOR {cell_id} REPORTING -> Stress Level: {physical_stress}")

        # B) Compose the transaction (Extrinsic)
        # We call the 'report_physical_state' function inside 'TemplateModule'
        call = substrate.compose_call(
            call_module='Template', 
            call_function='report_physical_state',
            call_params={
                'cell_id': cell_id,
                'stress': physical_stress
            }
        )

        # C) Sign and Submit the transaction
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
        
        print("   🚀 Submitting transaction to Layer-1...")
        
        # 'wait_for_inclusion=True' ensures we wait until the block is mined
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

        # D) Verification
        if receipt.is_success:
            print(f"   🔒 CONFIRMED IN BLOCK #{receipt.block_number}!")
            print(f"   Hash: {receipt.extrinsic_hash}")
            # Optional: Check if events were emitted
            for event in receipt.triggered_events:
                print(f"   ✨ Event Emitted: {event.value}")
        else:
            print(f"   ⚠️ Error: Transaction failed. Message: {receipt.error_message}")

        print("-" * 60)
        
        # Wait 6 seconds (standard block time) before the next reading
        time.sleep(6)

except KeyboardInterrupt:
    print("\n🛑 Oracle stopped manually.")