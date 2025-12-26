#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrionChain mesh simulator (20x20):
- LP market clearing with line limits
- Always-feasible via unmet (load shedding) + spill (curtailment)
- Line-limit shock on a vertical corridor
- Dynamic nodal price heuristic + heatmaps saved to PNG

Requires:
  pip install numpy scipy matplotlib
Run:
  python trionchain_lmp_20x20.py
Outputs:
  plots + prices_last.png, prices_mean.png, prices_std.png
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# -----------------------------
# Heatmap saver
# -----------------------------
def save_price_maps(prices_by_step: List[np.ndarray], W: int, H: int, out_prefix: str = "prices") -> None:
    """
    prices_by_step: list of arrays shape (N,), N=W*H
    Saves 3 PNG files:
      - prices_last.png (last step)
      - prices_mean.png (mean over time)
      - prices_std.png  (std / volatility over time)
    """
    P = np.stack(prices_by_step, axis=0)      # (T, N)
    P2 = P.reshape(P.shape[0], H, W)          # (T, H, W)

    last_map = P2[-1]
    mean_map = P2.mean(axis=0)
    std_map = P2.std(axis=0)

    def plot_map(M: np.ndarray, title: str, fname: str) -> None:
        plt.figure(figsize=(7, 6))
        plt.imshow(M, origin="lower", aspect="equal")
        plt.colorbar(label="Price")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(fname, dpi=160)
        plt.close()

    plot_map(last_map, "Price map (last step)", f"{out_prefix}_last.png")
    plot_map(mean_map, "Price map (mean over time)", f"{out_prefix}_mean.png")
    plot_map(std_map, "Price map (std over time)", f"{out_prefix}_std.png")


# -----------------------------
# Config
# -----------------------------
@dataclass
class SimConfig:
    W: int = 20
    H: int = 20
    T: int = 60
    seed: int = 7

    # Network
    base_edge_cap: float = 40.0  # MW per line (edge), before shock

    # Shock (vertical corridor between col=shock_col-1 and shock_col)
    shock_t0: int = 18
    shock_t1: int = 40
    shock_col: int = 10
    shock_factor: float = 0.25   # multiply capacity on corridor edges during shock

    # Node load/gen shaping
    base_load_mw: float = 2.2
    base_gen_cap_mw: float = 6.0

    # LP costs
    gen_cost: float = 10.0
    unmet_penalty: float = 10_000.0
    spill_penalty: float = 1.0

    # Price heuristic (scarcity + congestion)
    scarcity_price_k: float = 60.0
    congestion_price_k: float = 25.0


# -----------------------------
# Grid helpers
# -----------------------------
def node_id(x: int, y: int, W: int) -> int:
    return y * W + x

def id_xy(i: int, W: int) -> Tuple[int, int]:
    return (i % W, i // W)

def build_grid_edges(W: int, H: int) -> List[Tuple[int, int]]:
    """Undirected 4-neighbor edges, stored once (right and down)."""
    edges = []
    for y in range(H):
        for x in range(W):
            u = node_id(x, y, W)
            if x + 1 < W:
                v = node_id(x + 1, y, W)
                edges.append((u, v))
            if y + 1 < H:
                v = node_id(x, y + 1, W)
                edges.append((u, v))
    return edges


# -----------------------------
# Shocked capacities
# -----------------------------
def apply_line_limit_shock(
    W: int,
    edges: List[Tuple[int, int]],
    base_caps: np.ndarray,
    t: int,
    cfg: SimConfig,
) -> np.ndarray:
    """
    Reduce caps for edges crossing the vertical cut between col=shock_col-1 and shock_col
    during the shock window [shock_t0, shock_t1].
    """
    caps = base_caps.copy()
    if cfg.shock_t0 <= t <= cfg.shock_t1:
        c = cfg.shock_col
        for e, (u, v) in enumerate(edges):
            ux, uy = id_xy(u, W)
            vx, vy = id_xy(v, W)
            # horizontal edges crossing the cut
            if uy == vy and ((ux == c - 1 and vx == c) or (ux == c and vx == c - 1)):
                caps[e] = base_caps[e] * cfg.shock_factor
    return caps


# -----------------------------
# LP: flows + gen + unmet + spill
# -----------------------------
def solve_flow_lp(
    W: int,
    H: int,
    edges: List[Tuple[int, int]],
    edge_caps: np.ndarray,
    load: np.ndarray,
    gen_cap: np.ndarray,
    cfg: SimConfig,
) -> Dict[str, np.ndarray]:
    """
    Variables:
      g_i     generation >=0, <=gen_cap[i]
      u_i     unmet >=0, <=load[i]
      s_i     spill >=0, <=gen_cap[i]
      f_e     flow on edge e (signed), -cap<=f<=cap

    Balance per node:
      g_i + (sum_in - sum_out) + u_i - s_i = load_i
    """
    N = W * H
    E = len(edges)

    idx_g0 = 0
    idx_u0 = idx_g0 + N
    idx_s0 = idx_u0 + N
    idx_f0 = idx_s0 + N
    nvars = idx_f0 + E

    # Objective
    c = np.zeros(nvars, dtype=float)
    c[idx_g0:idx_g0 + N] = cfg.gen_cost
    c[idx_u0:idx_u0 + N] = cfg.unmet_penalty
    c[idx_s0:idx_s0 + N] = cfg.spill_penalty

    # Equality constraints
    A_eq = np.zeros((N, nvars), dtype=float)
    b_eq = load.astype(float).copy()

    # g + u - s
    for i in range(N):
        A_eq[i, idx_g0 + i] = 1.0
        A_eq[i, idx_u0 + i] = 1.0
        A_eq[i, idx_s0 + i] = -1.0

    # flows incidence (signed variable per edge)
    # f_e > 0 means u -> v
    for e, (u, v) in enumerate(edges):
        j = idx_f0 + e
        A_eq[u, j] += -1.0
        A_eq[v, j] += +1.0

    # Bounds
    bounds: List[Tuple[float, float]] = []
    bounds.extend([(0.0, float(gen_cap[i])) for i in range(N)])          # g
    bounds.extend([(0.0, float(load[i])) for i in range(N)])             # u
    bounds.extend([(0.0, float(gen_cap[i])) for i in range(N)])          # s
    bounds.extend([(-float(edge_caps[e]), float(edge_caps[e])) for e in range(E)])  # f

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x
    g = x[idx_g0:idx_g0 + N]
    u = x[idx_u0:idx_u0 + N]
    s = x[idx_s0:idx_s0 + N]
    f = x[idx_f0:idx_f0 + E]

    # Congestion
    util_edges = np.abs(f) / (edge_caps + 1e-12)
    avg_util = float(np.mean(util_edges))

    # Scarcity
    total_load = float(load.sum())
    unmet_ratio = float(u.sum()) / (total_load + 1e-12)

    # Simple nodal congestion proxy: average utilization incident to node
    node_util = np.zeros(N, dtype=float)
    deg = np.zeros(N, dtype=float)
    for e, (a, b) in enumerate(edges):
        ue = util_edges[e]
        node_util[a] += ue
        node_util[b] += ue
        deg[a] += 1.0
        deg[b] += 1.0
    node_util = node_util / (deg + 1e-12)

    # Nodal prices (heuristic, stable)
    prices = cfg.gen_cost + cfg.scarcity_price_k * unmet_ratio + cfg.congestion_price_k * node_util

    avg_price = float(cfg.gen_cost + cfg.scarcity_price_k * unmet_ratio + cfg.congestion_price_k * avg_util)

    return {
        "gen": g,
        "unmet": u,
        "spill": s,
        "flow": f,
        "avg_util": np.array([avg_util], dtype=float),
        "unmet_ratio": np.array([unmet_ratio], dtype=float),
        "prices": prices,
        "avg_price": np.array([avg_price], dtype=float),
    }


# -----------------------------
# Metrics
# -----------------------------
def boundary_mismatch(W: int, edges: List[Tuple[int, int]], flows: np.ndarray, shock_col: int) -> float:
    """Absolute net flow crossing the vertical cut between col=shock_col-1 and shock_col."""
    net = 0.0
    c = shock_col
    for e, (u, v) in enumerate(edges):
        ux, uy = id_xy(u, W)
        vx, vy = id_xy(v, W)
        if uy == vy and ((ux == c - 1 and vx == c) or (ux == c and vx == c - 1)):
            # f>0 means u->v
            if ux == c - 1 and vx == c:
                net += flows[e]
            else:
                net -= flows[e]
    return float(abs(net))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = SimConfig()
    rng = np.random.default_rng(cfg.seed)

    W, H, T = cfg.W, cfg.H, cfg.T
    N = W * H

    edges = build_grid_edges(W, H)
    E = len(edges)

    base_caps = np.full(E, cfg.base_edge_cap, dtype=float)

    # Spatial fields for load + gen capacity
    base_load_field = cfg.base_load_mw * (0.9 + 0.2 * rng.random(N))
    gen_cap_field = cfg.base_gen_cap_mw * (0.9 + 0.2 * rng.random(N))

    # Make left side slightly more generation to force corridor transfers
    for i in range(N):
        x, y = id_xy(i, W)
        if x < W // 2:
            gen_cap_field[i] *= 1.25

    # Time curves
    t_idx = np.arange(T)
    load_scale = 1.0 + 0.25 * np.sin(2 * np.pi * (t_idx / 32.0) + 1.0)
    gen_scale = 1.0 + 0.15 * np.sin(2 * np.pi * (t_idx / 32.0) - 0.5)

    # Inventory bookkeeping
    inventory = 0.0

    # Logs
    mismatch_log = []
    traded_log = []
    unmet_log = []
    spill_log = []
    avg_price_log = []
    p10_log = []
    p90_log = []
    util_log = []
    inv_log = []
    scarcity_log = []

    # Heatmaps log
    prices_by_step: List[np.ndarray] = []

    for tt in range(T):
        load = base_load_field * load_scale[tt]
        gen_cap = gen_cap_field * gen_scale[tt]
        caps = apply_line_limit_shock(W, edges, base_caps, tt, cfg)

        sol = solve_flow_lp(W, H, edges, caps, load, gen_cap, cfg)

        flows = sol["flow"]
        unmet = sol["unmet"]
        spill = sol["spill"]
        prices = sol["prices"]

        # Save prices for heatmaps
        prices_by_step.append(prices.copy())

        mismatch = boundary_mismatch(W, edges, flows, cfg.shock_col)

        total_load = float(load.sum())
        total_unmet = float(unmet.sum())
        total_spill = float(spill.sum())
        total_gen_used = float(sol["gen"].sum()) - total_spill
        served = total_load - total_unmet

        traded = max(0.0, served)  # proxy

        # Inventory: if you produced more than served, store it (simple 1h step)
        surplus = max(0.0, total_gen_used - served)
        inventory += surplus

        avg_price = float(sol["avg_price"][0])
        p10 = float(np.percentile(prices, 10))
        p90 = float(np.percentile(prices, 90))
        util = float(sol["avg_util"][0])
        scarcity = float(sol["unmet_ratio"][0])

        mismatch_log.append(mismatch)
        traded_log.append(traded)
        unmet_log.append(total_unmet)
        spill_log.append(total_spill)
        avg_price_log.append(avg_price)
        p10_log.append(p10)
        p90_log.append(p90)
        util_log.append(util)
        inv_log.append(inventory)
        scarcity_log.append(scarcity)

    # Save heatmaps
    save_price_maps(prices_by_step, W, H, out_prefix="prices")
    print("Saved heatmaps: prices_last.png, prices_mean.png, prices_std.png")

    # Plots
    x = np.arange(T)

    plt.figure()
    plt.plot(x, mismatch_log, marker="o")
    plt.title("Boundary Mismatch (Flows) – LINE LIMITS + Market (2-hop)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch (net flow across corridor)")
    plt.grid(True)

    plt.figure()
    plt.plot(x, traded_log, label="Traded MW")
    plt.plot(x, unmet_log, label="Unmet MW")
    plt.plot(x, spill_log, label="Spill MW")
    plt.title("Internal Market Clearing (2-hop, Line Limits)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, avg_price_log, label="Avg Price")
    plt.plot(x, p10_log, label="P10 Price")
    plt.plot(x, p90_log, label="P90 Price")
    plt.title("Prices (Dynamic + Congestion)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, util_log, label="Avg Edge Utilization")
    plt.title("Network Congestion (avg edge utilization)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Utilization")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, inv_log, label="Surplus Inventory (MWh)")
    plt.plot(x, scarcity_log, label="Global Scarcity (unmet/load)")
    plt.title("Inventory & Global Scarcity")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
