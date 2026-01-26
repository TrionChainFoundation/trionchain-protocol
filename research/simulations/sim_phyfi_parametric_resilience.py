import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SIMULACIÓN: TRIONCHAIN INDUSTRIAL DEFI (PHYFI)
# Archivo: sim_phyfi_parametric_resilience.py
# Escenario: "Stress-Linked Finance" & "Parametric Insurance"
# ---------------------------------------------------------

def run_phyfi_simulation(steps=100, seed=42):
    np.random.seed(seed)
    t = np.arange(steps)
    
    # --- 1. CAPA FÍSICA (FEM/SENSORS) ---
    # Simulamos condiciones ambientales (ej. Temperatura/Viento/Presión)
    # Una onda base + ruido + un evento extremo
    env_stress = 0.3 * np.sin(t * 0.1) + 0.4
    env_stress += np.random.normal(0, 0.05, steps)
    
    # EVENTO EXTREMO (El "Cisne Negro" físico) entre t=60 y t=75
    env_stress[60:75] += 0.5  # Pico de estrés físico repentino
    
    # Eficiencia del Activo (Physical Efficiency)
    # Cuando el estrés sube, la eficiencia baja (física básica)
    asset_efficiency = 1.0 - (env_stress * 0.8)
    asset_efficiency = np.clip(asset_efficiency, 0.1, 1.0)
    
    # --- 2. CAPA FINANCIERA (TRIONCHAIN SMART CONTRACTS) ---
    
    # PRODUCTO A: "Dynamic Rate Loan" (Préstamo de Tasa Dinámica)
    # Lógica: Si el activo es eficiente, pagas menos interés.
    # Si el activo está en riesgo (baja eficiencia), el riesgo sube -> tasa sube.
    base_rate = 0.05 # 5% base
    risk_premium_factor = 0.10
    loan_interest_rate = base_rate + (risk_premium_factor * (1.0 - asset_efficiency))
    
    # PRODUCTO B: "Parametric Insurance Payout" (Seguro Paramétrico)
    # Lógica: Si el estrés físico supera el umbral (0.75), el contrato paga automáticamente.
    # Trigger = "Hard Physics Limit"
    stress_threshold = 0.75
    insurance_payout = np.zeros(steps)
    
    # El contrato revisa la física en cada bloque
    for i in range(steps):
        if env_stress[i] > stress_threshold:
            # Payout proporcional a la gravedad del evento
            insurance_payout[i] = (env_stress[i] - stress_threshold) * 100000 # $$ ficticios
            
    # PRODUCTO C: "Tokenized Asset Value" (Valor del Activo en el Mercado)
    
    # Valor sin seguro (Naive Market) - Cae con la ineficiencia
    token_value_raw = 100 * asset_efficiency
    
    # Valor con cobertura TrionChain (Hedged Market)
    # El payout del seguro se inyecta en la liquidez del activo para compensar pérdidas
    token_value_hedged = token_value_raw + (insurance_payout / 2000) 

    return t, env_stress, loan_interest_rate, insurance_payout, token_value_raw, token_value_hedged, stress_threshold

# --- VISUALIZACIÓN ---
def plot_phyfi_results():
    plt.style.use('dark_background') # Estilo institucional "Deep Tech"
    
    t, stress, rates, payouts, val_raw, val_hedged, threshold = run_phyfi_simulation()
    
    # Aumentamos el tamaño vertical y definimos 'sharex' para compartir el eje de tiempo
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    
    # GRÁFICA 1: LA REALIDAD FÍSICA (La Causa)
    ax1.plot(t, stress, color='#00FFFF', linewidth=2, label='Physical Stress (FEM Input)')
    ax1.axhline(threshold, color='#FF0000', linestyle='--', linewidth=1.5, label='Insurance Trigger Threshold')
    ax1.fill_between(t, stress, threshold, where=(stress > threshold), color='#FF0000', alpha=0.3, label='Critical Event')
    ax1.set_ylabel('Physical Stress Index', fontweight='bold', fontsize=10)
    ax1.set_title('LAYER 1: PHYSICAL REALITY (Sensors/FEM)', fontsize=12, color='white', fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # GRÁFICA 2: LA REACCIÓN DEL SMART CONTRACT (El Mecanismo)
    color_rate = '#FFD700' # Gold
    ax2.plot(t, rates * 100, color=color_rate, label='Dynamic Interest Rate (%)')
    ax2.set_ylabel('Loan Interest Rate %', color=color_rate, fontweight='bold', fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color_rate)
    
    # Eje gemelo para el Payout (Barras Verdes)
    ax2b = ax2.twinx()
    color_payout = '#00FF00' # Green
    ax2b.bar(t, payouts, color=color_payout, alpha=0.6, width=1.0, label='Insurance Payout ($)')
    ax2b.set_ylabel('Automated Payout ($)', color=color_payout, fontweight='bold', fontsize=10)
    ax2b.tick_params(axis='y', labelcolor=color_payout)
    
    ax2.set_title('LAYER 2: AUTOMATED DEFI EXECUTION', fontsize=12, color='white', fontweight='bold', pad=10)
    
    # Unir leyendas de ambos ejes
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    # GRÁFICA 3: EL VALOR DEL ACTIVO (El Resultado Económico)
    ax3.plot(t, val_raw, color='gray', linestyle='--', label='Asset Value (Unprotected)')
    ax3.plot(t, val_hedged, color='white', linewidth=2.5, label='Asset Value (TrionChain Hedged)')
    
    ax3.set_ylabel('Tokenized Asset NAV', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Time (Simulation Steps)', fontweight='bold', fontsize=10)
    ax3.set_title('LAYER 3: ECONOMIC RESILIENCE', fontsize=12, color='white', fontweight='bold', pad=10)
    
    # ANOTACIÓN CORREGIDA: Posicionada para no chocar con el título
    ax3.annotate('Physics-Based Hedge\nProtects Value', 
                 xy=(67, val_hedged[67]),    # Dónde apunta la flecha (al valor protegido)
                 xytext=(45, 65),            # Dónde está el texto (Más abajo y a la izquierda)
                 arrowprops=dict(facecolor='white', shrink=0.05, width=1.5), 
                 color='white', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.8))
    
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.2)
    
    # AJUSTE FINAL DE ESPACIADO: Separa las gráficas verticalmente
    plt.subplots_adjust(hspace=0.4, top=0.93, bottom=0.07)
    
    plt.show()

if __name__ == "__main__":
    plot_phyfi_results()