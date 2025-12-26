import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from dataclasses import dataclass


# =========================
# CONFIG
# =========================

@dataclass
class Config:
    W: int = 20
    H: int = 20
    T: int = 60

    base_price: float = 40.0
    congestion_alpha: float = 8.0
    unmet_penalty: float = 200.0
    spill_penalty: float = 0.0      # puedes subirlo (ej 1.0) si quieres evitar spill excesivo

    line_limit: float = 120.0


cfg = Config()


# =========================
# GRID HELPERS
# =========================

def node_id(x: int, y: int, W: int) -> int:
    return y * W + x


def neighbors(x: int, y: int, W: int, H: int):
    nbrs = []
    if x > 0:
        nbrs.append((x - 1, y))
    if x < W - 1:
        nbrs.append((x + 1, y))
    if y > 0:
        nbrs.append((x, y - 1))
    if y < H - 1:
        nbrs.append((x, y + 1))
    return nbrs


# =========================
# LP SOLVER (with unmet + spill)
# =========================

def solve_flow_lp(W: int, H: int, load: np.ndarray, gen: np.ndarray,
                  line_limit: float, unmet_penalty: float, spill_penalty: float):
    """
    Vars:
      f_k   : signed flow per undirected edge, bounds [-L, +L]
      u_i   : unmet demand at node i, u_i >= 0
      s_i   : spill/curtailment at node i, s_i >= 0

    Balance per node i:
        net_in - net_out + gen_i - s_i + u_i = load_i

    This guarantees feasibility both for deficits (u_i) and surpluses (s_i).
    """

    N = W * H

    # undirected edges (i<j)
    edges = []
    for y in range(H):
        for x in range(W):
            i = node_id(x, y, W)
            for nx, ny in neighbors(x, y, W, H):
                j = node_id(nx, ny, W)
                if i < j:
                    edges.append((i, j))

    E = len(edges)

    # decision vector: [flows (E), unmet (N), spill (N)]
    nvar = E + N + N

    # objective
    c = np.zeros(nvar)
    c[E:E+N] = unmet_penalty
    c[E+N:] = spill_penalty

    # constraints: A_eq x = b_eq
    A_eq = np.zeros((N, nvar))
    b_eq = load.copy()

    # incidence for flows
    for k, (i, j) in enumerate(edges):
        # flow positive i -> j
        A_eq[i, k] -= 1.0
        A_eq[j, k] += 1.0

    # add unmet (+u_i)
    for i in range(N):
        A_eq[i, E + i] = 1.0

    # add spill (-s_i)
    for i in range(N):
        A_eq[i, E + N + i] = -1.0

    # move gen to LHS by subtracting it from RHS:
    # net + gen - spill + unmet = load  =>  net - spill + unmet = load - gen
    b_eq = load - gen

    # bounds
    bounds = []
    # flows
    bounds += [(-line_limit, line_limit) for _ in range(E)]
    # unmet: 0..load_i
    bounds += [(0.0, float(load[i])) for i in range(N)]
    # spill: 0..gen_i
    bounds += [(0.0, float(gen[i])) for i in range(N)]

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        raise RuntimeError(f"LP infeasible: {res.message}")

    flows = res.x[:E]
    unmet = res.x[E:E+N]
    spill = res.x[E+N:]
    return flows, unmet, spill, edges


# =========================
# SIMULATION
# =========================

def main():
    W, H, T = cfg.W, cfg.H, cfg.T
    N = W * H

    mismatch_hist = []
    traded_hist = []
    unmet_hist = []
    spill_hist = []
    price_hist = []
    congestion_hist = []
    inventory_hist = []
    scarcity_hist = []

    inventory = 0.0

    for t in range(T):

        # ---- synthetic load & gen ----
        load = 1.5 + 0.5 * np.sin(2 * np.pi * t / 40 + np.linspace(0, 3, N))
        load = np.clip(load, 0.1, None)

        g = 1.2 + 0.4 * np.sin(2 * np.pi * t / 35)
        gen = np.full(N, g)

        # scale
        load_mw = load * 400.0
        gen_mw = gen * 400.0

        # ---- solve ----
        flows, unmet, spill, edges = solve_flow_lp(
            W=W,
            H=H,
            load=load_mw,
            gen=gen_mw,
            line_limit=cfg.line_limit,
            unmet_penalty=cfg.unmet_penalty,
            spill_penalty=cfg.spill_penalty
        )

        # ---- metrics ----
        traded = float(np.sum(np.abs(flows)))
        unmet_total = float(np.sum(unmet))
        spill_total = float(np.sum(spill))

        mismatch = float(np.mean(unmet))  # avg unmet per node

        utilization = float(np.mean(np.abs(flows)) / cfg.line_limit)

        total_load = float(np.sum(load_mw))
        scarcity = (unmet_total / total_load) if total_load > 0 else 0.0

        price = (
            cfg.base_price
            + cfg.congestion_alpha * utilization
            + 50.0 * scarcity
        )

        # inventory: only real net surplus that is NOT spilled
        # produced - (served load) where served load = total_load - unmet_total
        served_load = total_load - unmet_total
        total_gen = float(np.sum(gen_mw)) - spill_total
        surplus = total_gen - served_load
        inventory += max(0.0, surplus)

        # ---- store ----
        mismatch_hist.append(mismatch)
        traded_hist.append(traded)
        unmet_hist.append(unmet_total)
        spill_hist.append(spill_total)
        price_hist.append(price)
        congestion_hist.append(utilization)
        inventory_hist.append(inventory)
        scarcity_hist.append(scarcity)

    # =========================
    # PLOTS
    # =========================
    ts = np.arange(T)

    plt.figure(figsize=(12, 6))
    plt.plot(ts, mismatch_hist, marker="o")
    plt.title("Boundary Mismatch (Flows) – LINE LIMITS + Market (2-hop)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch (avg unmet per node)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(ts, traded_hist, label="Traded MW")
    plt.plot(ts, unmet_hist, label="Unmet MW")
    plt.plot(ts, spill_hist, label="Spill MW")
    plt.title("Internal Market Clearing (2-hop, Line Limits)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(ts, price_hist, label="Avg Price")
    plt.title("Prices (Dynamic + Congestion)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(ts, congestion_hist, label="Avg Edge Utilization")
    plt.title("Network Congestion (avg edge utilization)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Utilization")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(ts, inventory_hist, label="Surplus Inventory (MWh)")
    plt.plot(ts, scarcity_hist, label="Global Scarcity (unmet/load)")
    plt.title("Inventory & Global Scarcity")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
