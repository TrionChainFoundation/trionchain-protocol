from substrateinterface import SubstrateInterface, Keypair
import time
import random

# --- CONFIGURACIÓN NACIONAL ---
REGIONS = [
    { "id": 10, "name": "PETRO CITY (Oil)", "capacity": 50000, "rate": 500, "unit": "bbl" },
    { "id": 11, "name": "AGRO VALLEY (Soy)", "capacity": 20000, "rate": 200, "unit": "tons" }
]

# Inventarios Iniciales
inventory = {10: 5000, 11: 2000}

print("🔌 Connecting to TrionChain (National Resource Ledger)...")
try:
    substrate = SubstrateInterface(url="ws://127.0.0.1:9944", type_registry_preset='substrate-node-template')
except:
    print("❌ Error: Node not running.")
    exit()

keypair = Keypair.create_from_uri('//Alice')
print("✅ Connected. Tracking Sovereign Commodities.")

# Registro Inicial
print("🔐 Registering Regions...")
nonce = substrate.get_account_nonce(keypair.ss58_address)
for region in REGIONS:
    call = substrate.compose_call('Template', 'register_sensor', {'cell_id': region['id'], 'sensor_account': keypair.ss58_address})
    substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce), wait_for_inclusion=False)
    nonce += 1
time.sleep(4)

try:
    while True:
        print("\n🔄 PRODUCTION CYCLE (Day +1)")
        # Obtenemos el nonce base para este bloque
        current_nonce = substrate.get_account_nonce(keypair.ss58_address)

        for i, r in enumerate(REGIONS):
            rid = r['id']
            
            # Lógica de Producción
            production = r['rate'] + random.randint(-50, 50)
            inventory[rid] += production
            
            fill_pct = (inventory[rid] / r['capacity']) * 100
            stress = int(fill_pct * 10)
            
            is_selling = 0
            if fill_pct >= 80: 
                print(f"   💰 EXPORT TRIGGERED for {r['name']}! Liquidity Released.")
                inventory[rid] = int(r['capacity'] * 0.2) 
                is_selling = 1 
                stress = 200

            print(f"   🏭 {r['name']} | Stock: {inventory[rid]} | Filled: {fill_pct:.1f}%")

            # Enviar a Blockchain
            call = substrate.compose_call(
                call_module='Template',
                call_function='report_state',
                call_params={
                    'cell_id': rid,
                    'stress': stress,
                    'generation': production,
                    'demand': is_selling,
                    'soc': int(fill_pct),
                    'price': inventory[rid]
                }
            )
            
            # --- CORRECCIÓN AQUÍ: SUMAMOS 'i' AL NONCE ---
            extrinsic = substrate.create_signed_extrinsic(
                call=call, 
                keypair=keypair, 
                nonce=current_nonce + i
            )
            
            substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
            
        time.sleep(6)

except KeyboardInterrupt:
    print("\n🛑 Stopped.")