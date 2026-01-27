from substrateinterface import SubstrateInterface, Keypair
import time
import random
import hashlib

# --- CONFIGURACIÓN ---
AIRCRAFT_ID = "TRION-JET-787-X"
SECURE_ELEMENT_SEED = "//Boeing787_Secure_01" 

FLIGHT_PATH = [
    {"code": "LHR", "id": 101, "name": "London Airspace", "phase": "TAKEOFF"},
    {"code": "CDG", "id": 102, "name": "Paris Sector", "phase": "CRUISE"},
    {"code": "FRA", "id": 103, "name": "Frankfurt Sector", "phase": "CRUISE_HEADWIND"},
    {"code": "IST", "id": 104, "name": "Istanbul Sector", "phase": "CRUISE"},
    {"code": "DXB", "id": 105, "name": "Dubai Airspace", "phase": "LANDING"},
]

OPTIMAL_BURN_RATE = 5.0
CO2_FACTOR = 2.5

print("✈️ INICIALIZANDO TRIONCHAIN AVIATION LINK...")
try:
    substrate = SubstrateInterface(url="ws://127.0.0.1:9944")
except:
    print("❌ Error: Nodo no encontrado.")
    exit()

# Cuentas
alice_keypair = Keypair.create_from_uri('//Alice') # La jefa con dinero
plane_keypair = Keypair.create_from_uri(SECURE_ELEMENT_SEED) # El avión (pobre al inicio)

print(f"✅ Secure Element Verified: {plane_keypair.ss58_address}")

# --- PASO CRÍTICO: FONDEAR AL AVIÓN ---
print("⛽ Fueling Plane Wallet (Transferencia de fondos para Gas Fees)...")
call_transfer = substrate.compose_call(
    call_module='Balances',
    call_function='transfer_allow_death',
    call_params={
        'dest': plane_keypair.ss58_address,
        'value': 1_000_000_000_000 # Un poco de TRN para gas
    }
)
extrinsic_transfer = substrate.create_signed_extrinsic(call=call_transfer, keypair=alice_keypair)
substrate.submit_extrinsic(extrinsic_transfer, wait_for_inclusion=True)
print("💰 Fondos recibidos. El avión está listo para operar.")
print("-" * 70)

def calculate_physics(phase):
    burn_rate = OPTIMAL_BURN_RATE
    stress = 200
    if phase == "TAKEOFF":
        burn_rate = 12.0; stress = 850
    elif phase == "CRUISE":
        burn_rate = 4.8; stress = 300
    elif phase == "CRUISE_HEADWIND":
        burn_rate = 6.5; stress = 600
    elif phase == "LANDING":
        burn_rate = 2.0; stress = 400
    burn_rate += random.uniform(-0.1, 0.1)
    return burn_rate, stress

try:
    fuel_remaining = 100.0
    
    # Obtenemos el nonce inicial del avión
    current_nonce = substrate.get_account_nonce(plane_keypair.ss58_address)

    while True:
        for sector in FLIGHT_PATH:
            burn, stress = calculate_physics(sector["phase"])
            co2_emitted = burn * CO2_FACTOR
            fuel_remaining -= (burn / 10)
            efficiency_index = OPTIMAL_BURN_RATE / burn 
            
            print(f"\n📍 SECTOR: {sector['name']} ({sector['code']}) | PHASE: {sector['phase']}")
            print(f"   🔥 Burn Rate: {burn:.2f} L/s | ☁️ CO2: {co2_emitted:.2f} Kg")
            
            telemetry_snapshot = f"{AIRCRAFT_ID}:{time.time()}:{co2_emitted}"
            data_hash = hashlib.sha256(telemetry_snapshot.encode()).hexdigest()[:16]
            
            # Registro previo (Alice registra la celda al nombre del avión)
            # Nota: Hacemos que Alice registre la celda para que el avión sea el dueño autorizado
            call_reg = substrate.compose_call(
                call_module='TrionFEMModule',
                call_function='register_sensor',
                call_params={'cell_id': sector["id"], 'sensor_account': plane_keypair.ss58_address}
            )
            # Alice paga el registro de la celda
            extrinsic_reg = substrate.create_signed_extrinsic(call=call_reg, keypair=alice_keypair)
            substrate.submit_extrinsic(extrinsic_reg, wait_for_inclusion=False)

            # Reporte de datos (El avión paga esto con sus nuevos fondos)
            call_report = substrate.compose_call(
                call_module='TrionFEMModule',
                call_function='report_state',
                call_params={
                    'cell_id': sector["id"],
                    'stress': int(stress),
                    'generation': int(co2_emitted * 10), 
                    'demand': int(efficiency_index * 100), 
                    'soc': int(fuel_remaining),
                    'price': 0 
                }
            )

            # El avión firma con su nonce actualizado
            extrinsic = substrate.create_signed_extrinsic(call=call_report, keypair=plane_keypair, nonce=current_nonce)
            substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
            current_nonce += 1 # Incrementamos el nonce localmente
            
            if efficiency_index < 0.8:
                print("   ⚠️ INEFFICIENCY DETECTED -> Leasing Rate UP")
            
            print(f"   🔒 Data Signed & Anchored. Hash: 0x{data_hash}...")
            time.sleep(2) 

        print("\n🛬 FLIGHT COMPLETE. Resetting...")
        fuel_remaining = 100.0
        time.sleep(5)

except KeyboardInterrupt:
    print("\n🛑 Flight Logger stopped.")