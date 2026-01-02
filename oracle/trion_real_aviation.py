import requests
import time
from substrateinterface import SubstrateInterface, Keypair

# --- REAL API CONFIGURATION (OpenSky Network) ---
# TARGET: EAST ASIA (China, Japan, Korea)
# Time: Friday Morning (Peak Traffic Guaranteed)
LAMIN = 20.0   
LAMAX = 45.0   
LOMIN = 110.0  
LOMAX = 145.0  

API_URL = f"https://opensky-network.org/api/states/all?lamin={LAMIN}&lamax={LAMAX}&lomin={LOMIN}&lomax={LOMAX}"

# --- BLOCKCHAIN CONNECTION ---
print("🔌 Connecting to TrionChain...")
try:
    substrate = SubstrateInterface(url="ws://127.0.0.1:9944", type_registry_preset='substrate-node-template')
except:
    print("❌ Error: Node not running.")
    exit()

keypair = Keypair.create_from_uri('//Alice')
print("✅ Connected. Scanning ASIA PACIFIC airspace.")
print(f"📡 Querying API: {API_URL}")
print("-" * 60)

def get_real_flights():
    try:
        response = requests.get(API_URL, timeout=15) # Más tiempo de espera
        data = response.json()
        states = data.get('states', [])
        if states is None: return [] # A veces la API devuelve None si está saturada
        return states
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return []

try:
    while True:
        print(f"\n📡 Ping OpenSky Satellites...")
        flights = get_real_flights()
        
        # DEBUG: Ver cuántos aviones encontró realmente
        print(f"   >>> RAW DATA: Found {len(flights)} objects in the sky.")

        if not flights:
            print("   No valid data returned. Waiting...")
        else:
            current_nonce = substrate.get_account_nonce(keypair.ss58_address)
            processed_count = 0
            
            for flight in flights:
                # Procesar máximo 4 aviones por ciclo para verlos bien
                if processed_count >= 4: break
                
                callsign = flight[1].strip() 
                velocity = flight[9] or 0    
                altitude = flight[7] or 0    
                
                # Filtro mínimo: Que tenga nombre y esté volando
                if callsign == "" or altitude < 100: continue 

                # PHYFI LOGIC
                cell_id = 100 + (hash(callsign) % 900) 
                if cell_id < 0: cell_id *= -1
                
                stress = int(velocity * 2) 
                if stress > 1000: stress = 999
                
                co2_proxy = int(altitude / 10) 

                print(f"✈️ FLIGHT: {callsign} | Vel: {velocity:.0f} m/s | Alt: {altitude:.0f}m")
                
                # Enviar a Blockchain
                call = substrate.compose_call(
                    call_module='Template',
                    call_function='report_state',
                    call_params={
                        'cell_id': cell_id,
                        'stress': stress,
                        'generation': co2_proxy, 
                        'demand': 100,           
                        'soc': 50,               
                        'price': 0
                    }
                )
                
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=keypair, nonce=current_nonce
                )
                substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
                
                print("   ✅ Anchored on-chain.")
                
                current_nonce += 1
                processed_count += 1
                time.sleep(0.1) 

        print("-" * 60)
        time.sleep(10) 

except KeyboardInterrupt:
    print("\n🛑 Stopped.")