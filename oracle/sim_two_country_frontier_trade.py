import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Two-Country TrionMesh Simulator
# -----------------------------
# Country A: exporter (left mesh)
# Country B: importer (right mesh)
#
# Key visuals:
# 1) Two-country price map (mean) with "hot frontier"
# 2) Two-country stress/volatility map (std)
# 3) Cross-border trade flow A→B (MWh per step)
# 4) Settlement cost paid by B in PAY token (per step)
# 5) Protocol fees paid in TRN token (per step)
# 6) Border corridor utilization heatmap (corridor x time)
#
# Border colored dots:
# - represent corridor interconnectors
# - color = utilization bucket (low/med/high/near-sat/sat)
#
# NOTE:
# This is a conceptual simulation to demonstrate TrionChain ideas:
# - Local domains (TrionCells)
# - Boundary constraints (corridors)
# - Emergent price stress at the frontier
# - Deterministic settlement and protocol fees (TRN)
#
# -----------------------------

def smooth2d(A, passes=1):
    """Simple neighbor averaging (cheap 'diffusion') to create realistic spatial continuity."""
    A = A.copy()
    for _ in range(passes):
        up = np.roll(A, -1, axis=0)
        dn = np.roll(A, 1, axis=0)
        lf = np.roll(A, 1, axis=1)
        rt = np.roll(A, -1, axis=1)
        A = 0.55 * A + 0.1125 * (up + dn + lf + rt)
    return A

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def build_base_fields(n, seed=7):
    """Generate spatial patterns for demand and generation."""
    rng = np.random.default_rng(seed)

    # Start with random noise, then smooth to look 'geographic'
    demand = rng.normal(1.0, 0.15, size=(n, n))
    gen    = rng.normal(1.0, 0.15, size=(n, n))

    demand = smooth2d(demand, passes=5)
    gen    = smooth2d(gen, passes=5)

    demand = demand / demand.mean()
    gen    = gen / gen.mean()
    return demand, gen

def time_profile(T, phase=0.0):
    """Global time variation (day/night style)."""
    t = np.arange(T)
    # Smooth deterministic oscillation
    return 1.0 + 0.25 * np.sin(2*np.pi*(t/T)*2 + phase) + 0.10 * np.sin(2*np.pi*(t/T)*5 + 0.7 + phase)

def pick_corridors(n, k=6, seed=3):
    """Select k frontier corridors as row indices."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(1, n-2, k, dtype=int)
    # small deterministic "shuffle"
    rng.shuffle(rows)
    return rows.tolist()

def utilization_color(u):
    """Bucket utilization u in [0,1] to a categorical color."""
    if u >= 0.98:
        return "#ff2d2d"  # saturated red
    if u >= 0.85:
        return "#ff7a00"  # orange
    if u >= 0.60:
        return "#ffd000"  # yellow
    if u >= 0.30:
        return "#42d86c"  # green
    return "#2ea7ff"      # blue

def run_two_country_sim(
    n=20,
    T=60,
    base_price_A=1.0,
    base_price_B=2.4,
    alpha_unmet=2.0,
    beta_cong=1.2,
    internal_capacity=0.35,
    corridor_capacity_base=0.22,
    fee_rate=0.032,          # 3.2% fee in TRN terms (conceptual)
    trn_price_in_pay=30.0,   # 1 TRN = 30 PAY (conceptual exchange rate for fee conversion)
    seed=11
):
    """
    Returns a dict with time-series + maps.
    """
    rng = np.random.default_rng(seed)

    # Base spatial fields
    demandA0, genA0 = build_base_fields(n, seed=seed+1)
    demandB0, genB0 = build_base_fields(n, seed=seed+2)

    # Make A an exporter: slightly higher gen, lower demand
    genA0 *= 1.10
    demandA0 *= 0.95

    # Make B an importer: slightly lower gen, higher demand
    genB0 *= 0.90
    demandB0 *= 1.07

    # Time variation
    profA = time_profile(T, phase=0.2)
    profB = time_profile(T, phase=1.1)

    # Frontier corridor definitions: A right edge connects to B left edge
    corridors = pick_corridors(n, k=6, seed=seed+3)

    # Corridor capacities can vary slightly by row (geographic constraints)
    corridor_caps = {}
    for r in corridors:
        corridor_caps[r] = corridor_capacity_base * rng.uniform(0.85, 1.15)

    # State arrays over time
    priceA_t = np.zeros((T, n, n))
    priceB_t = np.zeros((T, n, n))
    stressA_t = np.zeros((T, n, n))
    stressB_t = np.zeros((T, n, n))

    flow_AB = np.zeros(T)             # MWh per step (aggregate)
    settlement_pay = np.zeros(T)      # PAY token paid by B per step
    fee_trn = np.zeros(T)             # TRN fee paid per step
    corridor_util = np.zeros((len(corridors), T))  # utilization per corridor over time

    # Helper indices
    A_frontier_col = n - 1
    B_frontier_col = 0

    # Simulation
    for t in range(T):
        # Build time-varying supply/demand
        demandA = demandA0 * profA[t]
        genA    = genA0    * (1.0 + 0.20*np.cos(2*np.pi*(t/T)*2 + 0.5))

        demandB = demandB0 * profB[t]
        genB    = genB0    * (1.0 + 0.18*np.cos(2*np.pi*(t/T)*2 + 1.2))

        # Add tiny deterministic "weather-like" perturbation
        wobble = 0.03*np.sin(2*np.pi*t/17.0)
        genA *= (1.0 + wobble)
        genB *= (1.0 - 0.5*wobble)

        # Compute raw imbalance: positive = surplus, negative = deficit
        imbA = genA - demandA
        imbB = genB - demandB

        # INTERNAL balancing (cheap approximation):
        # Allow neighbor exchange limited by internal_capacity to reduce sharp discontinuities.
        # This acts like local rebalancing within each country.
        def internal_rebalance(imb, cap):
            # One pass of "redistribution" toward neighbors
            avg_nb = 0.25*(np.roll(imb,1,0)+np.roll(imb,-1,0)+np.roll(imb,1,1)+np.roll(imb,-1,1))
            delta = clamp(avg_nb - imb, -cap, cap)
            return imb + 0.6*delta

        imbA = internal_rebalance(imbA, internal_capacity)
        imbB = internal_rebalance(imbB, internal_capacity)

        # CROSS-BORDER trading:
        # Only along selected corridor rows. Energy flows from A frontier cell -> B frontier cell
        # if A has surplus and B has deficit. Limited by corridor capacity.
        step_flow = 0.0
        for i, r in enumerate(corridors):
            a_surplus = max(0.0, imbA[r, A_frontier_col])
            b_deficit = max(0.0, -imbB[r, B_frontier_col])

            cap = corridor_caps[r] * (1.0 + 0.15*np.sin(2*np.pi*t/29.0 + r*0.3))  # mild time variation
            cap = max(0.01, cap)

            traded = min(a_surplus, b_deficit, cap)

            # Apply trade to imbalances
            imbA[r, A_frontier_col] -= traded
            imbB[r, B_frontier_col] += traded

            step_flow += traded
            corridor_util[i, t] = traded / cap

        flow_AB[t] = step_flow

        # Prices (emergent):
        # - base_price + alpha*unmet_demand + beta*congestion/stress proxy
        unmetA = np.maximum(0.0, -imbA)
        unmetB = np.maximum(0.0, -imbB)

        # Stress proxy: magnitude of local imbalance + frontier penalty where corridors saturate
        stressA = np.abs(imbA)
        stressB = np.abs(imbB)

        # Frontier "hotline": add extra stress in B frontier when import corridors saturate
        for i, r in enumerate(corridors):
            u = corridor_util[i, t]
            # stress bump around the frontier cells (A right edge and B left edge)
            bump = 0.9 * (u**2)
            stressA[r, A_frontier_col] += 0.35 * bump
            stressB[r, B_frontier_col] += 1.25 * bump  # importer feels stronger frontier stress

        # Convert stress to "congestion term"
        congA = smooth2d(stressA, passes=2)
        congB = smooth2d(stressB, passes=2)

        priceA = base_price_A + alpha_unmet * smooth2d(unmetA, passes=1) + beta_cong * congA
        priceB = base_price_B + alpha_unmet * smooth2d(unmetB, passes=1) + beta_cong * congB

        # Normalize prices for nicer plots (optional, but keeps maps readable)
        # Keep them positive
        priceA = np.maximum(0.01, priceA)
        priceB = np.maximum(0.01, priceB)

        # Store
        priceA_t[t] = priceA
        priceB_t[t] = priceB
        stressA_t[t] = congA
        stressB_t[t] = congB

        # Settlement:
        # B pays in PAY token for imported energy. Use B frontier price as reference.
        # (Conceptual: settlement price could be corridor marginal price or agreed index.)
        if step_flow > 0:
            # weighted by frontier prices at corridor rows
            frontier_prices = []
            for r in corridors:
                frontier_prices.append(priceB[r, B_frontier_col])
            frontier_price = float(np.mean(frontier_prices))
        else:
            frontier_price = float(np.mean(priceB))

        settlement = step_flow * frontier_price
        settlement_pay[t] = settlement

        # Protocol fee paid in TRN:
        # fee_rate * settlement (in PAY) converted to TRN using exchange rate
        fee_pay = fee_rate * settlement
        fee_trn[t] = fee_pay / trn_price_in_pay

    return {
        "priceA_t": priceA_t,
        "priceB_t": priceB_t,
        "stressA_t": stressA_t,
        "stressB_t": stressB_t,
        "flow_AB": flow_AB,
        "settlement_pay": settlement_pay,
        "fee_trn": fee_trn,
        "corridors": corridors,
        "corridor_util": corridor_util,
    }

def plot_results(sim, n=20, title_prefix="Two-Country"):
    priceA_t = sim["priceA_t"]
    priceB_t = sim["priceB_t"]
    flow_AB = sim["flow_AB"]
    settlement_pay = sim["settlement_pay"]
    fee_trn = sim["fee_trn"]
    corridors = sim["corridors"]
    corridor_util = sim["corridor_util"]

    # Time aggregation
    priceA_mean = priceA_t.mean(axis=0)
    priceB_mean = priceB_t.mean(axis=0)
    priceA_std  = priceA_t.std(axis=0)
    priceB_std  = priceB_t.std(axis=0)

    # Build combined maps (A | frontier | B)
    # Use a 1-column "frontier separator" just for visuals
    frontier_col = np.full((n, 1), np.nan)

    mean_map = np.concatenate([priceA_mean, frontier_col, priceB_mean], axis=1)
    std_map  = np.concatenate([priceA_std,  frontier_col, priceB_std],  axis=1)

    # ---------- Figure 1: Mean price map with frontier markers ----------
    fig1 = plt.figure(figsize=(12, 6), dpi=160)
    ax1 = plt.gca()

    # Use a diverging colormap "coolwarm" to emphasize hot/cold regions
    im1 = ax1.imshow(mean_map, aspect="auto", cmap="coolwarm")

    ax1.set_title(f"{title_prefix} Price Map (Mean Over Time) — Frontier Stress", fontsize=14)
    ax1.set_xlabel("Country A  |  Frontier  |  Country B")
    ax1.set_ylabel("TrionCell row")

    # Draw frontier line (between A and separator)
    frontier_x = n - 0.5
    ax1.axvline(frontier_x, color="white", linewidth=2, alpha=0.9)

    # Plot corridor dots on frontier
    # dots are placed on the frontier line near the separator column
    T = corridor_util.shape[1]
    last_t = T - 1
    dot_x_A = n - 0.8
    dot_x_B = n + 0.8

    for i, r in enumerate(corridors):
        u = corridor_util[i, last_t]
        c = utilization_color(u)
        # two dots: one in A side, one in B side (same corridor)
        ax1.scatter([dot_x_A], [r], s=90, color=c, edgecolor="black", linewidth=0.6, zorder=5)
        ax1.scatter([dot_x_B], [r], s=90, color=c, edgecolor="black", linewidth=0.6, zorder=5)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Price / Stress (arbitrary units)")

    # Legend for corridor utilization
    legend_items = [
        ("<30% util", "#2ea7ff"),
        ("30–60% util", "#42d86c"),
        ("60–85% util", "#ffd000"),
        ("85–98% util", "#ff7a00"),
        (">=98% util", "#ff2d2d"),
    ]
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=col,
                      markeredgecolor="black", markersize=9, label=lab)
               for lab, col in legend_items]
    ax1.legend(handles=handles, loc="lower right", framealpha=0.85, title="Frontier corridor utilization\n(color = last step)")

    plt.tight_layout()

    # ---------- Figure 2: Std (volatility) map ----------
    fig2 = plt.figure(figsize=(12, 6), dpi=160)
    ax2 = plt.gca()

    im2 = ax2.imshow(std_map, aspect="auto", cmap="coolwarm")
    ax2.set_title(f"{title_prefix} Stress Map (Std Over Time) — Frontier Volatility", fontsize=14)
    ax2.set_xlabel("Country A  |  Frontier  |  Country B")
    ax2.set_ylabel("TrionCell row")

    ax2.axvline(frontier_x, color="white", linewidth=2, alpha=0.9)

    # corridor dots on std map too
    for i, r in enumerate(corridors):
        u = corridor_util[i, last_t]
        c = utilization_color(u)
        ax2.scatter([dot_x_A], [r], s=90, color=c, edgecolor="black", linewidth=0.6, zorder=5)
        ax2.scatter([dot_x_B], [r], s=90, color=c, edgecolor="black", linewidth=0.6, zorder=5)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Price / Stress (arbitrary units)")

    plt.tight_layout()

    # ---------- Figure 3: Cross-border flow ----------
    fig3 = plt.figure(figsize=(10, 4), dpi=160)
    ax3 = plt.gca()
    ax3.plot(flow_AB)
    ax3.set_title("Cross-border trade flow A→B (MWh per step)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("MWh")
    ax3.grid(alpha=0.25)
    plt.tight_layout()

    # ---------- Figure 4: Settlement cost (PAY) ----------
    fig4 = plt.figure(figsize=(10, 4), dpi=160)
    ax4 = plt.gca()
    ax4.plot(settlement_pay)
    ax4.set_title("Settlement cost paid by B (PAY token, per step)")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("PAY")
    ax4.grid(alpha=0.25)
    plt.tight_layout()

    # ---------- Figure 5: Protocol fees (TRN) ----------
    fig5 = plt.figure(figsize=(10, 4), dpi=160)
    ax5 = plt.gca()
    ax5.plot(fee_trn)
    ax5.set_title("Protocol fees paid in TRN (per step)")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("TRN")
    ax5.grid(alpha=0.25)
    plt.tight_layout()

    # ---------- Figure 6: Corridor utilization heatmap ----------
    fig6 = plt.figure(figsize=(10, 4.8), dpi=160)
    ax6 = plt.gca()
    im6 = ax6.imshow(corridor_util, aspect="auto", cmap="viridis")
    ax6.set_title("Border corridor utilization (each corridor over time)")
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Corridor index")
    cbar6 = plt.colorbar(im6, ax=ax6)
    cbar6.set_label("Utilization (0–1)")
    plt.tight_layout()

    plt.show()

def main():
    plt.style.use("default")  # You can switch to "dark_background" if you prefer

    sim = run_two_country_sim(
        n=20,
        T=60,
        base_price_A=0.9,
        base_price_B=2.2,
        alpha_unmet=2.2,
        beta_cong=1.0,
        internal_capacity=0.32,
        corridor_capacity_base=0.20,
        fee_rate=0.032,
        trn_price_in_pay=30.0,
        seed=11,
    )

    plot_results(sim, n=20, title_prefix="Two-Country")

if __name__ == "__main__":
    main()
