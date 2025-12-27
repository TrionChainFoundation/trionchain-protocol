from substrateinterface import SubstrateInterface, Keypair
import time
import random

# --- CONFIGURATION ---
NODE_URL = "ws://127.0.0.1:9944"
CELLS_TO_SIMULATE = [1, 2, 3, 4]  # Simulating a 4-node Mesh Grid

print("🔌 Connecting to TrionChain Master Node...")

try:
    substrate = SubstrateInterface(
        url=NODE_URL, 
        type_registry_preset='substrate-node-template'
    )
except ConnectionRefusedError:
    print("❌ Error: Connection refused. Please start the Rust node first.")
    exit()

keypair = Keypair.create_from_uri('//Alice')
print(f"✅ Connection established.")

# --- 🚀 SETUP & REGISTRATION (CRITICAL STEP) ---
print("\n🔐 REGISTERING SENSORS (Proof of Authority)...")
# Alice claims ownership of all 4 cells so she can write data to them
current_nonce = substrate.get_account_nonce(keypair.ss58_address)

batch_calls = []
for cell_id in CELLS_TO_SIMULATE:
    print(f"   - Registering Cell ID: {cell_id} to Alice...")
    call = substrate.compose_call(
        call_module='Template',
        call_function='register_sensor',
        call_params={'cell_id': cell_id, 'sensor_account': keypair.ss58_address}
    )
    # Enviamos una por una incrementando el nonce
    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=current_nonce)
    substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
    current_nonce += 1

print("✅ Registration signals sent. Waiting 6s for block finalization...")
time.sleep(6) # Damos tiempo a que la blockchain procese los registros
print("-" * 60)

# --- MAIN LOOP ---
try:
    while True:
        # Actualizamos el nonce para el nuevo bloque
        current_nonce = substrate.get_account_nonce(keypair.ss58_address)
        current_block = substrate.get_block_header()['header']['number']
        
        print(f"🔄 SYNC CYCLE (Block #{current_block})")
        
        for i, cell_id in enumerate(CELLS_TO_SIMULATE):
            
            # Simulation Logic
            if cell_id == 1: 
                stress = random.randint(200, 400)
                gen = random.randint(80, 100)
                demand = 10
                soc = 90
            elif cell_id == 2: 
                stress = random.randint(600, 950) 
                gen = 5
                demand = random.randint(80, 100)
                soc = 40
            elif cell_id == 3: 
                stress = random.randint(100, 300)
                gen = 0
                demand = 0
                soc = 0
            else: 
                stress = 50
                gen = 20
                demand = 20
                soc = random.randint(10, 95)

            call = substrate.compose_call(
                call_module='Template',
                call_function='report_state',
                call_params={
                    'cell_id': cell_id,
                    'stress': stress,
                    'generation': gen,
                    'demand': demand,
                    'soc': soc,
                    'price': random.randint(5, 50)
                }
            )

            # Usamos el nonce calculado para enviar 4 tx en el mismo bloque
            extrinsic = substrate.create_signed_extrinsic(
                call=call, 
                keypair=keypair,
                nonce=current_nonce + i 
            )
            
            substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
            print(f"   📡 Cell {cell_id} >> Stress: {stress} (Sent)")

        print("⏳ Waiting for next block cycle...")
        print("-" * 60)
        time.sleep(6)

except KeyboardInterrupt:
    print("\n🛑 Mesh Simulation stopped manually.")