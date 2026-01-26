from substrateinterface import SubstrateInterface, Keypair
import time
import random

# CONFIGURACIÓN
# ID 500 = Productor (Upstream)
# ID 501 = Inversor (Downstream/Custody)
PRODUCER_ID = 500
INVESTOR_ID = 501

OIL_PRICE = 75.50

print("🔌 Connecting to TrionChain (PhyFi Settlement Layer)...")
try:
    substrate = SubstrateInterface(url="ws://127.0.0.1:9944", type_registry_preset='substrate-node-template')
except:
    print("❌ Error: Node not running.")
    exit()

keypair = Keypair.create_from_uri('//Alice')
nonce = substrate.get_account_nonce(keypair.ss58_address)

# Registro de ambos actores
print("🔐 Registering Wallets & Sensors...")
for node_id in [PRODUCER_ID, INVESTOR_ID]:
    call = substrate.compose_call('Template', 'register_sensor', {'cell_id': node_id, 'sensor_account': keypair.ss58_address})
    substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce), wait_for_inclusion=False)
    nonce += 1
time.sleep(4)

# ESTADO INICIAL
producer_stock = 10000
producer_cap = 20000

investor_stock = 0
investor_cap = 100000 

try:
    while True:
        # 1. PRODUCCIÓN (Solo el productor genera)
        production = random.randint(800, 1500) 
        producer_stock += production
        
        # 2. CÁLCULO DE LLENADO
        fill_pct = int((producer_stock / producer_cap) * 100)
        
        # 3. LÓGICA DE TRANSFERENCIA (Smart Contract Trigger)
        is_transfer = 0
        transfer_amount = 0
        
        # Si el productor llega al 80%, vende al inversor
        if fill_pct >= 80:
            transfer_amount = 5000
            producer_stock -= transfer_amount
            investor_stock += transfer_amount 
            
            is_transfer = 1
            fill_pct = int((producer_stock / producer_cap) * 100) 
            
            print(f"   💸 DEAL EXECUTED: Moved {transfer_amount} bbls to Investor.")

        print(f"🛢️ PRODUCER: {producer_stock} | 🚢 INVESTOR: {investor_stock}")

        # 4. ENVIAR A LA CADENA (Dos transacciones con NONCE MANUAL)
        current_nonce = substrate.get_account_nonce(keypair.ss58_address)
        
        # A) Reporte Productor (Usa el nonce actual)
        call_p = substrate.compose_call('Template', 'report_state', {
            'cell_id': PRODUCER_ID,
            'stress': 0,
            'generation': producer_stock, 
            'demand': is_transfer, # 1 si hubo venta
            'soc': fill_pct,
            'price': int(transfer_amount * OIL_PRICE) 
        })
        substrate.submit_extrinsic(
            substrate.create_signed_extrinsic(call=call_p, keypair=keypair, nonce=current_nonce), 
            wait_for_inclusion=False
        )

        # B) Reporte Inversor (Usa nonce + 1)
        call_i = substrate.compose_call('Template', 'report_state', {
            'cell_id': INVESTOR_ID,
            'stress': 0,
            'generation': investor_stock, 
            'demand': 0,
            'soc': int((investor_stock/investor_cap)*100),
            'price': 0
        })
        
        # ¡AQUÍ ESTÁ LA CORRECCIÓN! Sumamos 1 al nonce
        substrate.submit_extrinsic(
            substrate.create_signed_extrinsic(call=call_i, keypair=keypair, nonce=current_nonce + 1), 
            wait_for_inclusion=False
        )
        
        time.sleep(6) 

except KeyboardInterrupt:
    print("\n🛑 Stopped.")