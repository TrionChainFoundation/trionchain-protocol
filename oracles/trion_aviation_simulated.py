from substrateinterface import SubstrateInterface, Keypair
import time
import random

# --- RUTA REALISTA (EK202: JFK -> DXB) ---
ROUTE = [
    {"id": 101, "loc": "🇬🇧 UK (London)", "wind": 0},
    {"id": 102, "loc": "🇫🇷 FRANCE (Paris)", "wind": 0},
    {"id": 103, "loc": "🇩🇪 GERMANY (Frankfurt)", "wind": 10},
    {"id": 104, "loc": "🇹🇷 TURKEY (Istanbul)", "wind": 0},
    {"id": 200, "loc": "🇮🇶 IRAQ (Baghdad)", "wind": 5},
    {"id": 105, "loc": "🇦🇪 UAE (Dubai)", "wind": 0},
]

print("🔌 Connecting to TrionChain (Server Mode)...")
try:
    substrate = SubstrateInterface(url="ws://127.0.0.1:9944", type_registry_preset='substrate-node-template')
except:
    print("❌ Error: Node not running.")
    exit()

keypair = Keypair.create_from_uri('//Alice')
print("✅ Connected. Starting Loop.")

# 1. OBTENER EL NONCE INICIAL
print("🔐 Registering Path...")
current_nonce = substrate.get_account_nonce(keypair.ss58_address)

# Registro
for sector in ROUTE:
    call = substrate.compose_call('Template', 'register_sensor', {'cell_id': sector['id'], 'sensor_account': keypair.ss58_address})
    substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=current_nonce), wait_for_inclusion=False)
    current_nonce += 1
    
time.sleep(4) 

try:
    while True:
        print("\n🎬 ACTION: FLIGHT EK202 TAKEOFF")
        total_co2_accumulated = 0.0
        fuel = 100
        
        for sector in ROUTE:
            print(f"\n📍 NOW ENTERING -> {sector['loc']}")
            
            for _ in range(3): # 3 bloques por país
                velocity = random.randint(245, 255)
                stress = int(velocity * 1.5) + (sector['wind'] * 25) 
                burn_rate = 40 + sector['wind']
                total_co2_accumulated += burn_rate
                fuel -= 1
                
                print(f"   ✈️ Sending... Stress: {stress} | CO2: {total_co2_accumulated} | Nonce: {current_nonce}")

                call = substrate.compose_call(
                    call_module='Template',
                    call_function='report_state',
                    call_params={
                        'cell_id': sector['id'],
                        'stress': stress,
                        'generation': int(burn_rate * 10),
                        'demand': 1,
                        'soc': int(fuel),
                        'price': int(total_co2_accumulated)
                    }
                )
                
                # USAMOS EL CONTADOR MANUAL
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, 
                    keypair=keypair, 
                    nonce=current_nonce
                )
                
                substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
                
                current_nonce += 1
                time.sleep(6) 

except KeyboardInterrupt:
    print("\n🛑 Stopped.")
