#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrionChain Simulator (20x20) - Market Clearing with Line Limits + Battery (Scenario 1)
Battery reduces UNMET by discharging during scarcity and charging during spill (when available).

Runs two simulations:
  A) Baseline (no battery)
  B) With battery
and plots comparisons.

Dependencies:
  - numpy
  - scipy
  - matplotlib

Run:
  python trionchain_lmp_20x20_battery_unmet.py
Options:
  python trionchain_lmp_20x20_battery_unmet.py --steps 60 --grid 20 --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linprog


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class GridConfig:
    n: int = 20                  # grid size n x n
    steps: int = 60              # simulation steps (blocks)
    dt: float = 1.0              # time step duration (hours equivalent)
    line_capacity: float = 120.0 # MW per line
    flow_cost: float = 0.0005    # small cost to discourage cycling
    spill_cost: float = 0.05     # small penalty for spill
    unmet_penalty: float = 1000.0# large penalty for unmet (forces feasibility)
    gen_cost_base: float = 10.0  # base generation cost
    gen_cost_slope: float = 8.0  # cost variation across space
    gen_cap_base: float = 80.0   # base generation capacity MW
    gen_cap_var: float = 60.0    # spatial capacity variation
    load_base: float = 60.0      # base load MW
    load_var: float = 50.0       # spatial load variation
    demand_wave_amp: float = 0.35# temporal wave amplitude
    noise_sigma: float = 0.06    # temporal noise scale
    seed: int = 7


@dataclass
class BatteryConfig:
    enabled: bool = True
    # Battery locations: list of node indices in flattened (0..N-1)
    nodes: List[int] = None
    e_max_mwh: float = 1800.0     # energy capacity (MWh)
    p_max_mw: float = 280.0       # charge/discharge power limit (MW)
    eta_c: float = 0.95           # charge efficiency
    eta_d: float = 0.95           # discharge efficiency
    soc_init: float = 0.25        # initial SoC fraction


# ----------------------------
# Helper functions
# ----------------------------

def idx(r: int, c: int, n: int) -> int:
    return r * n + c


def build_grid_edges(n: int) -> List[Tuple[int, int]]:
    """
    Undirected neighbor edges (right + down). We'll model each as one flow variable (signed).
    """
    edges = []
    for r in range(n):
        for c in range(n):
            u = idx(r, c, n)
            if c + 1 < n:
                v = idx(r, c + 1, n)
                edges.append((u, v))
            if r + 1 < n:
                v = idx(r + 1, c, n)
                edges.append((u, v))
    return edges


def corridor_edges(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    Choose a "corridor cut" to compute boundary mismatch.
    We'll use a vertical cut between columns n//2 - 1 and n//2.
    Return indices of edges crossing that cut (u in left, v in right).
    """
    cut_col_left = (n // 2) - 1
    cut_col_right = n // 2
    left_set = set(idx(r, cut_col_left, n) for r in range(n))
    right_set = set(idx(r, cut_col_right, n) for r in range(n))

    cross = []
    for ei, (u, v) in enumerate(edges):
        if (u in left_set and v in right_set) or (v in left_set and u in right_set):
            cross.append(ei)
    return cross


def make_spatial_fields(cfg: GridConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create spatially varying:
      - gen_cost[i]
      - gen_cap[i]
      - base_load[i]
    """
    n = cfg.n
    N = n * n

    # Coordinates normalized (0..1)
    rr, cc = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    rr = rr.reshape(-1)
    cc = cc.reshape(-1)

    # Generator cost: gradient + slight noise
    gen_cost = cfg.gen_cost_base + cfg.gen_cost_slope * (0.6 * rr + 0.4 * cc)
    gen_cost += rng.normal(0, 0.6, size=N)
    gen_cost = np.clip(gen_cost, 3.0, None)

    # Gen capacity: higher on one side (think renewables region)
    gen_cap = cfg.gen_cap_base + cfg.gen_cap_var * (0.8 * (1 - rr) + 0.2 * cc)
    gen_cap += rng.normal(0, 6.0, size=N)
    gen_cap = np.clip(gen_cap, 0.0, None)

    # Load: higher toward bottom-right
    base_load = cfg.load_base + cfg.load_var * (0.55 * rr + 0.45 * cc)
    base_load += rng.normal(0, 5.0, size=N)
    base_load = np.clip(base_load, 0.0, None)

    return gen_cost, gen_cap, base_load


def demand_profile(cfg: GridConfig, t: int) -> float:
    """
    A smooth time-varying demand factor.
    """
    x = 2 * np.pi * t / max(1, cfg.steps - 1)
    # two waves to create multi-peak behavior
    return 1.0 + cfg.demand_wave_amp * (0.7 * np.sin(x) + 0.3 * np.sin(2.1 * x + 0.9))


# ----------------------------
# LP Market Clearing (with line limits)
# ----------------------------

@dataclass
class StepResult:
    gen: np.ndarray
    flows: np.ndarray
    unmet: np.ndarray
    spill: np.ndarray
    lmp: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray


def solve_step_lp(
    cfg: GridConfig,
    edges: List[Tuple[int, int]],
    gen_cost: np.ndarray,
    gen_cap: np.ndarray,
    load: np.ndarray,
    soc_prev: np.ndarray,
    bat: BatteryConfig,
) -> StepResult:
    """
    Solve one-step linear program with:
      - generation
      - flows (signed, bounded by line capacity)
      - unmet >=0
      - spill >=0
      - battery charge/discharge (if enabled), limited by soc and emax

    Nodal balance for each node i:
        gen_i - load_i + sum_in(flow) - sum_out(flow)
        + discharge_i - charge_i + unmet_i - spill_i = 0

    Objective:
        sum(gen_cost_i * gen_i) + unmet_penalty*sum(unmet_i) + spill_cost*sum(spill_i)
        + flow_cost*sum(|flow_e|)
        + tiny cost for charge/discharge to avoid degeneracy
    """
    n = cfg.n
    N = n * n
    E = len(edges)

    # Battery nodes mask / mapping
    if bat.enabled and bat.nodes:
        bat_nodes = list(bat.nodes)
    else:
        bat_nodes = []

    B = len(bat_nodes)

    # Variable layout:
    # [gen(N), flow_pos(E), flow_neg(E), unmet(N), spill(N), charge(B), discharge(B)]
    # flow = flow_pos - flow_neg, with each >=0 and <=cap
    # This keeps everything linear and bounded.
    offset = {}
    k = 0
    offset["gen"] = (k, k + N); k += N
    offset["fpos"] = (k, k + E); k += E
    offset["fneg"] = (k, k + E); k += E
    offset["unmet"] = (k, k + N); k += N
    offset["spill"] = (k, k + N); k += N
    offset["chg"] = (k, k + B); k += B
    offset["dis"] = (k, k + B); k += B
    nvar = k

    # Objective coefficients
    c = np.zeros(nvar)
    c[offset["gen"][0]:offset["gen"][1]] = gen_cost
    c[offset["unmet"][0]:offset["unmet"][1]] = cfg.unmet_penalty
    c[offset["spill"][0]:offset["spill"][1]] = cfg.spill_cost
    c[offset["fpos"][0]:offset["fpos"][1]] = cfg.flow_cost
    c[offset["fneg"][0]:offset["fneg"][1]] = cfg.flow_cost
    # tiny regularization
    if B > 0:
        c[offset["chg"][0]:offset["chg"][1]] = 0.001
        c[offset["dis"][0]:offset["dis"][1]] = 0.001

    # Bounds
    bounds = []

    # gen bounds
    for i in range(N):
        bounds.append((0.0, float(gen_cap[i])))

    # flow pos/neg bounds
    for _ in range(E):
        bounds.append((0.0, cfg.line_capacity))
    for _ in range(E):
        bounds.append((0.0, cfg.line_capacity))

    # unmet/spill bounds
    for _ in range(N):
        bounds.append((0.0, None))  # unmet
    for _ in range(N):
        bounds.append((0.0, None))  # spill

    # battery bounds based on soc
    # discharge <= soc_prev * eta_d / dt
    # charge <= (Emax - soc_prev) / (eta_c * dt)
    if B > 0:
        for bi, node in enumerate(bat_nodes):
            soc = float(soc_prev[bi])
            chg_max = min(bat.p_max_mw, (bat.e_max_mwh - soc) / (bat.eta_c * cfg.dt) if bat.eta_c > 0 else 0.0)
            dis_max = min(bat.p_max_mw, (soc * bat.eta_d) / cfg.dt if bat.eta_d > 0 else 0.0)
            chg_max = max(0.0, chg_max)
            dis_max = max(0.0, dis_max)
            bounds.append((0.0, chg_max))  # charge
        for bi, node in enumerate(bat_nodes):
            soc = float(soc_prev[bi])
            dis_max = min(bat.p_max_mw, (soc * bat.eta_d) / cfg.dt if bat.eta_d > 0 else 0.0)
            dis_max = max(0.0, dis_max)
            bounds.append((0.0, dis_max))  # discharge

    # Equality constraints: nodal balance for each node
    Aeq = np.zeros((N, nvar))
    beq = np.zeros(N)

    # gen contribution
    Aeq[:, offset["gen"][0]:offset["gen"][1]] = np.eye(N)

    # unmet/spill
    Aeq[:, offset["unmet"][0]:offset["unmet"][1]] = np.eye(N)
    Aeq[:, offset["spill"][0]:offset["spill"][1]] = -np.eye(N)

    # flows incidence
    # For edge e=(u,v), define flow = fpos_e - fneg_e, oriented u->v in incidence.
    # Node u: -flow, Node v: +flow  (since balance is gen - load + inflow - outflow ...)
    # We'll encode: for node u, subtract (fpos - fneg), for node v, add (fpos - fneg)
    fpos0, fpos1 = offset["fpos"]
    fneg0, fneg1 = offset["fneg"]

    for e, (u, v) in enumerate(edges):
        # u row
        Aeq[u, fpos0 + e] += -1.0
        Aeq[u, fneg0 + e] += +1.0
        # v row
        Aeq[v, fpos0 + e] += +1.0
        Aeq[v, fneg0 + e] += -1.0

    # battery (if enabled)
    if B > 0:
        chg0, chg1 = offset["chg"]
        dis0, dis1 = offset["dis"]
        for bi, node in enumerate(bat_nodes):
            Aeq[node, chg0 + bi] += -1.0  # charging consumes power
            Aeq[node, dis0 + bi] += +1.0  # discharging supplies power

    # RHS = load
    beq[:] = load

    # Solve LP
    res = linprog(
        c=c,
        A_eq=Aeq,
        b_eq=beq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"LP infeasible or failed: {res.message}")

    x = res.x

    gen = x[offset["gen"][0]:offset["gen"][1]]
    fpos = x[offset["fpos"][0]:offset["fpos"][1]]
    fneg = x[offset["fneg"][0]:offset["fneg"][1]]
    flows = fpos - fneg
    unmet = x[offset["unmet"][0]:offset["unmet"][1]]
    spill = x[offset["spill"][0]:offset["spill"][1]]

    if B > 0:
        charge = x[offset["chg"][0]:offset["chg"][1]]
        discharge = x[offset["dis"][0]:offset["dis"][1]]
    else:
        charge = np.zeros(0)
        discharge = np.zeros(0)

    # LMP approximation via duals of balance constraints (HiGHS marginals)
    # SciPy provides res.eqlin.marginals for equality constraints.
    lmp = np.zeros(N)
    try:
        lmp = np.array(res.eqlin.marginals, dtype=float)
    except Exception:
        # fallback: if unavailable, use gen_cost as proxy (not ideal)
        lmp = gen_cost.copy()

    # Update SOC externally (energy state)
    soc_next = soc_prev.copy()
    if B > 0:
        # soc_{t+1} = soc_t + eta_c*charge*dt - (discharge/eta_d)*dt
        soc_next = soc_prev + (bat.eta_c * charge * cfg.dt) - ((discharge / bat.eta_d) * cfg.dt)

        # clamp
        soc_next = np.clip(soc_next, 0.0, bat.e_max_mwh)

    return StepResult(
        gen=gen,
        flows=flows,
        unmet=unmet,
        spill=spill,
        lmp=lmp,
        charge=charge,
        discharge=discharge,
        soc=soc_next,
    )


# ----------------------------
# Simulation runner
# ----------------------------

@dataclass
class SimOutputs:
    traded_mw: np.ndarray
    unmet_mw: np.ndarray
    spill_mw: np.ndarray
    congestion: np.ndarray
    avg_price: np.ndarray
    p10_price: np.ndarray
    p90_price: np.ndarray
    mismatch: np.ndarray
    soc_trace: np.ndarray  # [steps, B]
    charge_trace: np.ndarray
    discharge_trace: np.ndarray


def run_simulation(cfg: GridConfig, bat: BatteryConfig) -> SimOutputs:
    rng = np.random.default_rng(cfg.seed)

    n = cfg.n
    N = n * n

    edges = build_grid_edges(n)
    E = len(edges)
    cross = corridor_edges(n, edges)

    gen_cost, gen_cap, base_load = make_spatial_fields(cfg, rng)

    # Battery nodes default: a small cluster around center
    if bat.enabled:
        if not bat.nodes:
            cr = n // 2
            cc = n // 2
            bat.nodes = [
                idx(cr, cc, n),
                idx(cr, cc - 1, n),
                idx(cr - 1, cc, n),
                idx(cr - 1, cc - 1, n),
            ]
    else:
        bat.nodes = []

    B = len(bat.nodes) if (bat.enabled and bat.nodes) else 0

    # init SoC
    soc = np.full(B, bat.soc_init * bat.e_max_mwh, dtype=float)

    traded_mw = np.zeros(cfg.steps)
    unmet_mw = np.zeros(cfg.steps)
    spill_mw = np.zeros(cfg.steps)
    congestion = np.zeros(cfg.steps)
    avg_price = np.zeros(cfg.steps)
    p10_price = np.zeros(cfg.steps)
    p90_price = np.zeros(cfg.steps)
    mismatch = np.zeros(cfg.steps)

    soc_trace = np.zeros((cfg.steps, B))
    chg_trace = np.zeros((cfg.steps, B))
    dis_trace = np.zeros((cfg.steps, B))

    # time loop
    for t in range(cfg.steps):
        # build load for this step
        scale = demand_profile(cfg, t)
        noise = rng.normal(0, cfg.noise_sigma, size=N)
        load = base_load * scale * (1.0 + noise)
        load = np.clip(load, 0.0, None)

        step = solve_step_lp(
            cfg=cfg,
            edges=edges,
            gen_cost=gen_cost,
            gen_cap=gen_cap,
            load=load,
            soc_prev=soc,
            bat=bat if bat.enabled else BatteryConfig(enabled=False, nodes=[]),
        )

        # Metrics
        unmet_mw[t] = float(np.sum(step.unmet))
        spill_mw[t] = float(np.sum(step.spill))

        # "Traded MW" = sum of positive absolute flows (proxy of network usage)
        traded_mw[t] = float(np.sum(np.abs(step.flows)))

        # Congestion = average utilization |f|/cap
        congestion[t] = float(np.mean(np.abs(step.flows) / cfg.line_capacity))

        # Prices
        avg_price[t] = float(np.mean(step.lmp))
        p10_price[t] = float(np.percentile(step.lmp, 10))
        p90_price[t] = float(np.percentile(step.lmp, 90))

        # Boundary mismatch: net flow across corridor cut (absolute net across cut)
        # For each crossing edge, define sign so left->right is positive net.
        net = 0.0
        left_col = (n // 2) - 1
        right_col = n // 2
        left_set = set(idx(r, left_col, n) for r in range(n))
        right_set = set(idx(r, right_col, n) for r in range(n))

        for ei in cross:
            u, v = edges[ei]
            f = step.flows[ei]
            if (u in left_set and v in right_set):
                net += f
            elif (v in left_set and u in right_set):
                net -= f
        mismatch[t] = abs(net)

        # Battery traces
        if B > 0:
            soc = step.soc
            soc_trace[t, :] = soc
            chg_trace[t, :] = step.charge
            dis_trace[t, :] = step.discharge

    return SimOutputs(
        traded_mw=traded_mw,
        unmet_mw=unmet_mw,
        spill_mw=spill_mw,
        congestion=congestion,
        avg_price=avg_price,
        p10_price=p10_price,
        p90_price=p90_price,
        mismatch=mismatch,
        soc_trace=soc_trace,
        charge_trace=chg_trace,
        discharge_trace=dis_trace,
    )


# ----------------------------
# Plotting
# ----------------------------

def plot_results(cfg: GridConfig, base: SimOutputs, batt: SimOutputs, bat_cfg: BatteryConfig) -> None:
    steps = cfg.steps
    x = np.arange(steps)

    # 1) Market clearing: Unmet / Spill (baseline vs battery)
    plt.figure()
    plt.plot(x, base.unmet_mw, label="Unmet (baseline)")
    plt.plot(x, batt.unmet_mw, label="Unmet (with battery)")
    plt.plot(x, base.spill_mw, label="Spill (baseline)")
    plt.plot(x, batt.spill_mw, label="Spill (with battery)")
    plt.title("Internal Market Clearing - Unmet & Spill (baseline vs battery)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()

    # 2) Prices
    plt.figure()
    plt.plot(x, base.avg_price, label="Avg Price (baseline)")
    plt.plot(x, batt.avg_price, label="Avg Price (with battery)")
    plt.plot(x, base.p10_price, label="P10 (baseline)")
    plt.plot(x, batt.p10_price, label="P10 (with battery)")
    plt.plot(x, base.p90_price, label="P90 (baseline)")
    plt.plot(x, batt.p90_price, label="P90 (with battery)")
    plt.title("Prices (Dynamic + Congestion) - baseline vs battery")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Price (dual units)")
    plt.legend()
    plt.tight_layout()

    # 3) Congestion
    plt.figure()
    plt.plot(x, base.congestion, label="Congestion (baseline)")
    plt.plot(x, batt.congestion, label="Congestion (with battery)")
    plt.title("Network Congestion (avg edge utilization) - baseline vs battery")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Utilization")
    plt.legend()
    plt.tight_layout()

    # 4) Boundary mismatch
    plt.figure()
    plt.plot(x, base.mismatch, label="Mismatch (baseline)")
    plt.plot(x, batt.mismatch, label="Mismatch (with battery)")
    plt.title("Boundary Mismatch (Flows) - LINE LIMITS + Market (corridor cut)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch (|net flow across corridor|)")
    plt.legend()
    plt.tight_layout()

    # 5) Battery SoC & actions
    if bat_cfg.enabled and bat_cfg.nodes:
        plt.figure()
        soc = batt.soc_trace
        if soc.size > 0:
            plt.plot(x, np.mean(soc, axis=1), label="Avg Battery SoC (MWh)")
            plt.title("Battery State of Charge (SoC)")
            plt.xlabel("Simulation step (block)")
            plt.ylabel("MWh")
            plt.legend()
            plt.tight_layout()

        plt.figure()
        chg = batt.charge_trace
        dis = batt.discharge_trace
        if chg.size > 0 and dis.size > 0:
            plt.plot(x, np.sum(chg, axis=1), label="Total Charge (MW)")
            plt.plot(x, np.sum(dis, axis=1), label="Total Discharge (MW)")
            plt.title("Battery Charge/Discharge Power")
            plt.xlabel("Simulation step (block)")
            plt.ylabel("MW")
            plt.legend()
            plt.tight_layout()

    plt.show()


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--line_capacity", type=float, default=120.0)
    parser.add_argument("--unmet_penalty", type=float, default=1000.0)
    parser.add_argument("--spill_cost", type=float, default=0.05)

    # Battery params
    parser.add_argument("--battery", action="store_true", help="Enable battery (default in this scenario)")
    parser.add_argument("--e_max", type=float, default=1800.0)
    parser.add_argument("--p_max", type=float, default=280.0)
    parser.add_argument("--soc_init", type=float, default=0.25)
    parser.add_argument("--eta_c", type=float, default=0.95)
    parser.add_argument("--eta_d", type=float, default=0.95)

    args = parser.parse_args()

    cfg = GridConfig(
        n=args.grid,
        steps=args.steps,
        seed=args.seed,
        line_capacity=args.line_capacity,
        unmet_penalty=args.unmet_penalty,
        spill_cost=args.spill_cost,
    )

    # Baseline: battery disabled
    bat_off = BatteryConfig(enabled=False, nodes=[])

    # Battery enabled (Scenario 1)
    bat_on = BatteryConfig(
        enabled=True if args.battery or True else False,  # keep enabled for scenario
        nodes=None,
        e_max_mwh=args.e_max,
        p_max_mw=args.p_max,
        soc_init=args.soc_init,
        eta_c=args.eta_c,
        eta_d=args.eta_d,
    )

    print("Running baseline (no battery)...")
    base = run_simulation(cfg, bat_off)

    print("Running with battery (reduce unmet)...")
    batt = run_simulation(cfg, bat_on)

    # Quick summary
    print("\n=== Summary ===")
    print(f"Total UNMET baseline : {base.unmet_mw.sum():.2f} MW-steps")
    print(f"Total UNMET battery  : {batt.unmet_mw.sum():.2f} MW-steps")
    print(f"UNMET reduction      : {(1 - (batt.unmet_mw.sum() / max(1e-9, base.unmet_mw.sum()))) * 100:.1f}%")
    print(f"Total SPILL baseline : {base.spill_mw.sum():.2f} MW-steps")
    print(f"Total SPILL battery  : {batt.spill_mw.sum():.2f} MW-steps")
    print(f"Avg congestion base  : {base.congestion.mean():.3f}")
    print(f"Avg congestion batt  : {batt.congestion.mean():.3f}")

    plot_results(cfg, base, batt, bat_on)


if __name__ == "__main__":
    main()
