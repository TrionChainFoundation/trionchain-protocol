# sim_crossborder_market_trn_fee.py
# Two-country (A|B) mesh simulation + frontier corridor dots WITH LEGEND
#
# Run:
#   python3 sim_crossborder_market_trn_fee.py
#
# Outputs (PNG) to:
#   simulations/energy-mesh/figures/

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize


# -----------------------------
# Helpers
# -----------------------------
def idx(i, j, n):
    return i * n + j


def build_grid_edges(n, cap_internal):
    edges = []
    for i in range(n):
        for j in range(n):
            u = idx(i, j, n)
            if i + 1 < n:
                v = idx(i + 1, j, n)
                edges.append((u, v, cap_internal))
            if j + 1 < n:
                v = idx(i, j + 1, n)
                edges.append((u, v, cap_internal))
    return edges


def make_border_corridors(n, k_corridors, cap_border, seed=7):
    rng = np.random.default_rng(seed)
    rows = np.linspace(0, n - 1, k_corridors, dtype=int)
    rows = np.clip(rows + rng.integers(-1, 2, size=k_corridors), 0, n - 1)
    rows = np.unique(rows)
    while len(rows) < k_corridors:
        rows = np.unique(np.append(rows, (len(rows) * 3) % n))
    rows = rows[:k_corridors]

    corridors = []
    for r in rows:
        a_u = idx(r, n - 1, n)  # east edge of A
        b_u = idx(r, 0, n)      # west edge of B
        corridors.append((a_u, b_u, cap_border))
    return corridors, rows


def diffuse_transfer(n, net, edges, n_iters=25, relax=0.35):
    recv = np.zeros_like(net, dtype=float)
    send = np.zeros_like(net, dtype=float)
    edge_flow = np.zeros(len(edges), dtype=float)
    x = net.astype(float).copy()

    for _ in range(n_iters):
        for e_i, (u, v, cap) in enumerate(edges):
            xu, xv = x[u], x[v]
            if xu > 0 and xv < 0:
                amt = min(cap, xu, -xv)
                x[u] -= amt
                x[v] += amt
                send[u] += amt
                recv[v] += amt
                edge_flow[e_i] += amt
            elif xv > 0 and xu < 0:
                amt = min(cap, xv, -xu)
                x[v] -= amt
                x[u] += amt
                send[v] += amt
                recv[u] += amt
                edge_flow[e_i] += amt

        if relax > 0:
            grid = x.reshape(n, n)
            avg = grid.copy()
            avg[1:-1, 1:-1] = (
                grid[1:-1, 1:-1]
                + grid[0:-2, 1:-1] + grid[2:, 1:-1]
                + grid[1:-1, 0:-2] + grid[1:-1, 2:]
            ) / 5.0
            x = (1 - relax) * x + relax * avg.reshape(-1)

    spill = np.clip(x, 0, None)
    unmet = np.clip(-x, 0, None)
    delivered = np.minimum(recv, np.clip(-net, 0, None))

    util = np.zeros(len(edges), dtype=float)
    for e_i, (_, _, cap) in enumerate(edges):
        util[e_i] = min(1.0, edge_flow[e_i] / (cap * max(1, n_iters)))
    return delivered, spill, unmet, util


def compute_price(base, unmet, spill, local_util_proxy, scarcity_scale=1.0):
    w_unmet = 2.5
    w_cong = 1.8
    w_spill = 0.6
    p = base + w_unmet * (unmet * scarcity_scale) + w_cong * local_util_proxy - w_spill * (spill * 0.3)
    return np.clip(p, 0.01, None)


def save_fig(path, dpi=220):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


# -----------------------------
# Simulation
# -----------------------------
def run_sim(
    n=20,
    T=60,
    cap_internal_A=2.0,
    cap_internal_B=2.0,
    k_corridors=6,
    cap_border=8.0,
    seed=42,
    base_price_A=1.0,
    base_price_B=1.0,
    fee_trn_per_mwh=0.015,
    fee_trn_fixed=0.25,
):
    rng = np.random.default_rng(seed)
    N = n * n

    edges_A = build_grid_edges(n, cap_internal_A)
    edges_B = build_grid_edges(n, cap_internal_B)

    corridors, corridor_rows = make_border_corridors(n, k_corridors, cap_border, seed=seed + 7)

    # Base generation/demand templates
    genA0 = 1.3 + 0.5 * rng.random(N)
    demA0 = 1.0 + 0.5 * rng.random(N)

    genB0 = 0.9 + 0.4 * rng.random(N)
    demB0 = 1.3 + 0.6 * rng.random(N)

    # A: stronger generation near export side (east border)
    for i in range(n):
        for j in range(n - 4, n):
            genA0[idx(i, j, n)] *= 1.25

    # B: stronger demand near import side (west border)
    for i in range(n):
        for j in range(0, 4):
            demB0[idx(i, j, n)] *= 1.25

    t = np.arange(T)
    day = 0.15 * np.sin(2 * np.pi * t / T)
    shock = np.zeros(T)
    shock[T // 2:T // 2 + 6] = 0.10

    priceA_hist = np.zeros((T, N))
    priceB_hist = np.zeros((T, N))

    trade_mwh_hist = np.zeros(T)
    trn_fee_hist = np.zeros(T)
    pay_cost_hist = np.zeros(T)

    corridor_util_hist = np.zeros((T, len(corridors)))  # utilization per corridor per step

    for tt in range(T):
        genA = genA0 * (1.05 + 0.05 * np.cos(2 * np.pi * tt / T))
        demA = demA0 * (1.00 + day[tt])

        genB = genB0 * (0.98 + 0.06 * np.cos(2 * np.pi * (tt + 7) / T))
        demB = demB0 * (1.05 + day[tt] + shock[tt])

        netA = genA - demA
        netB = genB - demB

        _, spillA, unmetA, utilA = diffuse_transfer(n, netA, edges_A, n_iters=20, relax=0.25)
        _, spillB, unmetB, utilB = diffuse_transfer(n, netB, edges_B, n_iters=20, relax=0.25)

        # congestion proxy per node
        congA = np.zeros(N); degA = np.zeros(N)
        for (u, v, _), uval in zip(edges_A, utilA):
            congA[u] += uval; congA[v] += uval
            degA[u] += 1; degA[v] += 1
        congA = congA / np.maximum(1, degA)

        congB = np.zeros(N); degB = np.zeros(N)
        for (u, v, _), uval in zip(edges_B, utilB):
            congB[u] += uval; congB[v] += uval
            degB[u] += 1; degB[v] += 1
        congB = congB / np.maximum(1, degB)

        priceA = compute_price(base_price_A, unmetA, spillA, congA)
        priceB = compute_price(base_price_B, unmetB, spillB, congB)

        traded_total = 0.0
        pay_cost = 0.0

        for c_i, (a_u, b_u, cap) in enumerate(corridors):
            avail = spillA[a_u]
            need = unmetB[b_u]

            if avail <= 0 or need <= 0:
                corridor_util_hist[tt, c_i] = 0.0
                continue

            flow = min(cap, avail, need)
            if flow <= 0:
                corridor_util_hist[tt, c_i] = 0.0
                continue

            spillA[a_u] -= flow
            unmetB[b_u] -= flow

            corridor_util_hist[tt, c_i] = flow / cap

            p_clear = 0.5 * (priceA[a_u] + priceB[b_u])
            pay_cost += flow * p_clear
            traded_total += flow

        trn_fee = (fee_trn_per_mwh * traded_total + fee_trn_fixed) if traded_total > 0 else 0.0

        # recompute after trade impacts
        priceA = compute_price(base_price_A, unmetA, spillA, congA)
        priceB = compute_price(base_price_B, unmetB, spillB, congB)

        priceA_hist[tt] = priceA
        priceB_hist[tt] = priceB
        trade_mwh_hist[tt] = traded_total
        trn_fee_hist[tt] = trn_fee
        pay_cost_hist[tt] = pay_cost

    return {
        "n": n,
        "T": T,
        "priceA_hist": priceA_hist,
        "priceB_hist": priceB_hist,
        "trade_mwh_hist": trade_mwh_hist,
        "trn_fee_hist": trn_fee_hist,
        "pay_cost_hist": pay_cost_hist,
        "corridor_util_hist": corridor_util_hist,
        "corridor_rows": corridor_rows,
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_two_country_heatmap_with_corridor_legend(
    priceA_hist,
    priceB_hist,
    n,
    corridor_rows,
    corridor_util_hist,
    mode="mean",
    out_path="simulations/energy-mesh/figures/two_country_frontier_heatmap_mean.png",
):
    """
    Combined heatmap: [Country A | Country B], frontier seam line, and colored dots on the seam.
    Dot color encodes *corridor stress* using mean utilization over time.
    Includes a legend + colorbar for the dot encoding.
    """
    if mode == "mean":
        A = priceA_hist.mean(axis=0).reshape(n, n)
        B = priceB_hist.mean(axis=0).reshape(n, n)
        title = "Two-Country Price Map (Mean Over Time) — Frontier Stress"
    elif mode == "std":
        A = priceA_hist.std(axis=0).reshape(n, n)
        B = priceB_hist.std(axis=0).reshape(n, n)
        title = "Two-Country Stress Map (Std Over Time) — Frontier Volatility"
    elif mode == "last":
        A = priceA_hist[-1].reshape(n, n)
        B = priceB_hist[-1].reshape(n, n)
        title = "Two-Country Price Map (Last Step) — Frontier State"
    else:
        raise ValueError("mode must be one of: mean, std, last")

    AB = np.hstack([A, B])

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(AB, cmap="coolwarm")
    fig.colorbar(im, ax=ax, label="Price / Stress (arbitrary units)")

    ax.set_title(title)
    ax.set_xlabel("Country A  |  Frontier  |  Country B")
    ax.set_ylabel("TrionCell row")

    # Frontier seam between A and B
    seam_x = n - 0.5
    ax.axvline(x=seam_x, linewidth=3)

    # Corridor stress = mean utilization over time for each corridor
    corridor_stress = corridor_util_hist.mean(axis=0)  # shape: (n_corridors,)
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Draw colored dots at seam (one dot per corridor row)
    sc = ax.scatter(
        [n - 1] * len(corridor_rows),  # A side seam column index
        corridor_rows,
        c=corridor_stress,
        cmap="viridis",
        norm=norm,
        s=80,
        edgecolors="black",
        linewidths=0.4,
        zorder=5,
    )
    # Optionally mirror on B side too:
    ax.scatter(
        [n] * len(corridor_rows),      # B side seam column index
        corridor_rows,
        c=corridor_stress,
        cmap="viridis",
        norm=norm,
        s=80,
        edgecolors="black",
        linewidths=0.4,
        zorder=5,
    )

    # Dot colorbar (corridor stress)
    cbar2 = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    cbar2.set_label("Frontier corridor stress (mean utilization, 0..1)")

    # A clean legend explaining what dots are
    dot_handle = Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Colored dots = active cross-border corridors (TrionBoundaries)\nColor intensity = corridor stress / utilization",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=9,
        linewidth=0
    )
    ax.legend(handles=[dot_handle], loc="upper left", frameon=True)

    plt.tight_layout()
    save_fig(out_path)
    plt.show()


def plot_timeseries(res, out_dir="simulations/energy-mesh/figures"):
    T = res["T"]
    t = np.arange(T)

    plt.figure()
    plt.plot(t, res["trade_mwh_hist"])
    plt.title("Cross-border trade flow A→B (MWh per step)")
    plt.xlabel("Step")
    plt.ylabel("MWh")
    plt.tight_layout()
    save_fig(f"{out_dir}/trade_flow_mwh.png")

    plt.figure()
    plt.plot(t, res["pay_cost_hist"])
    plt.title("Settlement cost paid by B (PAY token, per step)")
    plt.xlabel("Step")
    plt.ylabel("PAY")
    plt.tight_layout()
    save_fig(f"{out_dir}/settlement_cost_pay.png")

    plt.figure()
    plt.plot(t, res["trn_fee_hist"])
    plt.title("Protocol fees paid in TRN (per step)")
    plt.xlabel("Step")
    plt.ylabel("TRN")
    plt.tight_layout()
    save_fig(f"{out_dir}/protocol_fees_trn.png")

    util = res["corridor_util_hist"]
    plt.figure()
    plt.imshow(util.T, aspect="auto", cmap="viridis")
    plt.title("Border corridor utilization (each corridor over time)")
    plt.xlabel("Step")
    plt.ylabel("Corridor index")
    plt.colorbar(label="Utilization (0..1)")
    plt.tight_layout()
    save_fig(f"{out_dir}/border_corridor_utilization.png")


def main():
    out_dir = "simulations/energy-mesh/figures"

    res = run_sim(
        n=20,
        T=60,
        cap_internal_A=2.0,
        cap_internal_B=2.0,
        k_corridors=6,
        cap_border=8.0,
        seed=42,
        base_price_A=1.0,
        base_price_B=1.0,
        fee_trn_per_mwh=0.015,
        fee_trn_fixed=0.25,
    )

    # Combined frontier maps WITH corridor-dot legend
    plot_two_country_heatmap_with_corridor_legend(
        res["priceA_hist"], res["priceB_hist"], res["n"],
        corridor_rows=res["corridor_rows"],
        corridor_util_hist=res["corridor_util_hist"],
        mode="mean",
        out_path=f"{out_dir}/two_country_frontier_heatmap_mean.png",
    )

    plot_two_country_heatmap_with_corridor_legend(
        res["priceA_hist"], res["priceB_hist"], res["n"],
        corridor_rows=res["corridor_rows"],
        corridor_util_hist=res["corridor_util_hist"],
        mode="std",
        out_path=f"{out_dir}/two_country_frontier_heatmap_std.png",
    )

    # Time series + corridor utilization heatmap
    plot_timeseries(res, out_dir=out_dir)

    print(f"Saved figures to: {out_dir}/")
    plt.show()


if __name__ == "__main__":
    main()
