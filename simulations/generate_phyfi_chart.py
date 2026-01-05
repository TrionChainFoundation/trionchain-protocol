import numpy as np
import matplotlib.pyplot as plt

def plot_phyfi_chart():
    # Estilo "TrionChain Dark Mode"
    plt.style.use('dark_background')
    
    # Datos simulados
    t = np.arange(100)
    
    # 1. Física (Stress)
    stress = 0.3 * np.sin(t * 0.1) + 0.4 + np.random.normal(0, 0.02, 100)
    stress[60:75] += 0.5 # Evento crítico
    threshold = 0.75
    
    # 2. Finanzas (Payout)
    payout = np.zeros(100)
    payout[stress > threshold] = 40000 # Dinero liberado
    
    # 3. Valor del Activo
    val_raw = 100 - (stress * 20) # Cae con el estrés
    val_hedged = val_raw.copy()
    val_hedged[60:] += 35 # Se recupera gracias al seguro
    
    # CREAR GRÁFICA
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # LAYER 1: PHYSICS
    ax1.plot(t, stress, color='#00FFFF', linewidth=3, label='Physical Stress (Sensor)')
    ax1.axhline(threshold, color='#FF3333', linestyle='--', linewidth=2, label='Safety Threshold')
    ax1.fill_between(t, stress, threshold, where=(stress > threshold), color='#FF3333', alpha=0.3)
    ax1.set_ylabel('Physical Stress', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_title('LAYER 1: PHYSICAL REALITY (FEM Consensus)', color='white', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.1)
    
    # LAYER 2: DEFI EXECUTION
    ax2.bar(t, payout, color='#00FF00', alpha=0.8, width=1.0, label='Smart Contract Payout ($)')
    ax2.set_ylabel('Liquidity ($)', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.set_title('LAYER 2: AUTOMATED PHYFI EXECUTION', color='white', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.1)
    
    # LAYER 3: ECONOMIC RESULT
    ax3.plot(t, val_raw, color='gray', linestyle='--', linewidth=2, label='Value (Unprotected)')
    ax3.plot(t, val_hedged, color='white', linewidth=4, label='Value (TrionChain Hedged)')
    ax3.set_ylabel('Asset Value (NAV)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Time (Blocks)', fontweight='bold')
    
    # Flecha explicativa
    ax3.annotate('Financial Immunization', xy=(70, val_hedged[70]), xytext=(40, 90),
                 arrowprops=dict(facecolor='#00FF00', shrink=0.05), color='#00FF00', fontsize=12, fontweight='bold')
    
    ax3.legend(loc='lower left')
    ax3.set_title('LAYER 3: ECONOMIC RESILIENCE', color='white', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_phyfi_chart()