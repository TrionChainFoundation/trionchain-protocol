from substrateinterface import SubstrateInterface, Keypair
import time
import random

# 1. CONEXIÓN
print("🔌 Connecting to TrionChain (Institutional Node)...")
try:
    substrate = SubstrateInterface(
        url="ws://127.0.0.1:9944",
        
    )
except ConnectionRefusedError:
    print("❌ Error: Node not running. Please start the Substrate node first.")
    exit()

# 2. IDENTIDAD (Alice - Superuser)
keypair = Keypair.create_from_uri('//Alice')
cell_id = 1

print(f"✅ Connected as: {keypair.ss58_address}")

# 3. FASE DE REGISTRO (Seguridad)
# La nueva lógica exige que el sensor esté autorizado antes de enviar datos.
print("🔐 Registering Sensor on-chain...")

call_register = substrate.compose_call(
    call_module='TrionFEMModule',
    call_function='register_sensor',
    call_params={'cell_id': cell_id, 'sensor_account': keypair.ss58_address}
)

extrinsic_reg = substrate.create_signed_extrinsic(call=call_register, keypair=keypair)
receipt_reg = substrate.submit_extrinsic(extrinsic_reg, wait_for_inclusion=True)

if receipt_reg.is_success:
    print("✅ Sensor Authorized Successfully!")
else:
    print("⚠️ Sensor registration warning (maybe already registered).")

print("-" * 60)

# 4. BUCLE DE DATOS COMPLEJOS (Vector de Estado)
try:
    while True:
        # Generamos datos físicos simulados
        stress_val = random.randint(200, 800)
        generation_val = random.randint(50, 100)
        demand_val = random.randint(40, 90)
        soc_val = random.randint(20, 95)
        price_val = random.randint(10, 50)
        
        print(f"📡 SENDING PHYSICAL VECTOR -> Gen: {generation_val}MW | Soc: {soc_val}% | Stress: {stress_val}")

        # Llamada a la NUEVA función 'report_state' con todos los parámetros
        call = substrate.compose_call(
            call_module='TrionFEMModule',
            call_function='report_state',
            call_params={
                'cell_id': cell_id,
                'stress': stress_val,
                'generation': generation_val,
                'demand': demand_val,
                'soc': soc_val,
                'price': price_val
            }
        )

        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
        
        # Enviar y esperar
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

        if receipt.is_success:
            print(f"   🧱 Confirmed in Block #{receipt.block_number}")
            # Verificamos los eventos emitidos
            for event in receipt.triggered_events:
                print(f"   ✨ {event.value['event_id']}: {event.params}")
        else:
            print(f"   ❌ Error: {receipt.error_message}")

        print("-" * 60)
        time.sleep(6)

except KeyboardInterrupt:
    print("\n🛑 Oracle stopped.")