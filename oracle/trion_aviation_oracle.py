from substrateinterface import SubstrateInterface, Keypair
import time
import random
import hashlib

# --- CONFIGURACIÓN DE LA FLOTA (TrionObjects) ---
# Simulamos un Boeing 787 con un elemento seguro de hardware (Secure Element)
AIRCRAFT_ID = "TRION-JET-787-X"
SECURE_ELEMENT_SEED = "//Boeing787_Secure_01" 

# --- CONFIGURACIÓN DE LA RUTA (TrionCells) ---
# Mapeamos códigos IATA a IDs de celdas numéricos para la blockchain
FLIGHT_PATH = [
    {"code": "LHR", "id": 101, "name": "London Airspace", "phase": "TAKEOFF"},
    {"code": "CDG", "id": 102, "name": "Paris Sector", "phase": "CRUISE"},
    {"code": "FRA", "id": 103, "name": "Frankfurt Sector", "phase": "CRUISE_HEADWIND"}, # Viento en contra
    {"code": "IST", "id": 104, "name": "Istanbul Sector", "phase": "CRUISE"},
    {"code": "DXB", "id": 105, "name": "Dubai Airspace", "phase": "LANDING"},
]

# --- PARÁMETROS FÍSICOS DE REFERENCIA (Baseline) ---
OPTIMAL_BURN_RATE = 5.0  # Litros/segundo
CO2_FACTOR = 2.5         # Kg CO2 por Litro

# Conexión
print("✈️ INICIALIZANDO TRIONCHAIN AVIATION LINK...")
try:
    substrate = SubstrateInterface(url="ws://127.0.0.1:9944", type_registry_preset='substrate-node-template')
except:
    print("❌ Error: Nodo no encontrado. Asegúrate de que la blockchain esté corriendo.")
    exit()

# La llave del avión (El dispositivo IoT)
plane_keypair = Keypair.create_from_uri(SECURE_ELEMENT_SEED)
print(f"✅ Secure Element Verified: {plane_keypair.ss58_address}")
print(f"✅ Flight Plan Loaded: {AIRCRAFT_ID} :: LHR -> DXB")
print("-" * 70)

def calculate_physics(phase):
    """Simula la física real del motor según la fase de vuelo"""
    burn_rate = OPTIMAL_BURN_RATE
    stress = 200 # Normal
    
    if phase == "TAKEOFF":
        burn_rate = 12.0 # Alto consumo
        stress = 850     # Alto estrés mecánico
    elif phase == "CRUISE":
        burn_rate = 4.8  # Eficiente
        stress = 300
    elif phase == "CRUISE_HEADWIND":
        burn_rate = 6.5  # Ineficiente por clima
        stress = 600     # Turbulencia
    elif phase == "LANDING":
        burn_rate = 2.0
        stress = 400
        
    # Añadir variabilidad natural del sensor
    burn_rate += random.uniform(-0.1, 0.1)
    
    return burn_rate, stress

try:
    # Simulamos el vuelo paso a paso
    fuel_remaining = 100.0 # Porcentaje
    
    while True:
        # Recorremos los sectores aéreos (TrionCells)
        for sector in FLIGHT_PATH:
            
            # 1. OBTENER DATOS FÍSICOS (Nivel B - Signed Operational Data)
            burn, stress = calculate_physics(sector["phase"])
            co2_emitted = burn * CO2_FACTOR
            fuel_remaining -= (burn / 10) # Simulación de gasto
            
            # Cálculo de Eficiencia para PhyFi (Dynamic Leasing)
            # Si efficiency < 1.0, el avión está gastando más de lo debido -> Paga más leasing
            efficiency_index = OPTIMAL_BURN_RATE / burn 
            
            print(f"\n📍 SECTOR: {sector['name']} ({sector['code']}) | PHASE: {sector['phase']}")
            print(f"   🔥 Burn Rate: {burn:.2f} L/s | ☁️ CO2: {co2_emitted:.2f} Kg")
            
            # 2. REGISTRO PREVIO (Necesario para la demo si la celda no tiene dueño)
            # En producción esto se hace una vez, aquí lo aseguramos para que no falle el script
            call_reg = substrate.compose_call(
                call_module='Template',
                call_function='register_sensor',
                call_params={'cell_id': sector["id"], 'sensor_account': plane_keypair.ss58_address}
            )
            # Enviamos sin esperar para ir rápido, asumimos éxito
            substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=call_reg, keypair=plane_keypair), wait_for_inclusion=False)

            # 3. ENVIAR A TRIONCHAIN
            # Mapeamos los datos de aviación a la estructura existente del Pallet
            # Cell_ID = Sector Aéreo
            # Stress = Estrés del Motor
            # Generation = Emisión de CO2 (Lo usamos para guardar este dato visualmente)
            # Demand = Eficiencia * 100
            # SoC = Combustible Restante
            
            call = substrate.compose_call(
                call_module='Template',
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

            # FIRMA DEL DISPOSITIVO SEGURO
            extrinsic = substrate.create_signed_extrinsic(call=call, keypair=plane_keypair)
            receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
            
            if receipt.is_success:
                 print(f"   🔒 Data Signed & Anchored in Block #{receipt.block_number}")
            
            # 4. SIMULACIÓN DE IMPACTO PHYFI
            if efficiency_index < 0.8:
                print("   ⚠️ INEFFICIENCY DETECTED -> Smart Contract increases Leasing Rate (+0.5%)")
            
            # Tiempo de viaje entre sectores
            time.sleep(2) 

        print("\n🛬 FLIGHT COMPLETE. Resetting simulation loop...")
        fuel_remaining = 100.0
        time.sleep(5)

except KeyboardInterrupt:
    print("\n🛑 Flight Logger stopped.")