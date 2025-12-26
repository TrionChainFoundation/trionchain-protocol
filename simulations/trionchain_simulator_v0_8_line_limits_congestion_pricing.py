import math
import random
import json
import meshio
import matplotlib.pyplot as plt

EPS = 1e-12

# ============================================================
#  TRIONCHAIN SIMULATOR v0.8
#  FEM Mesh + Flows + LINE LIMITS + Internal Market (2-hop)
#  + Dynamic Pricing + Congestion Pricing (capacity-aware)
# ============================================================

class TrionCell:
    def __init__(self, eid, node_indices, coords):
        self.id = eid
        self.node_indices = node_indices
        self.coords = coords

        # drivers
        self.mineral_volume = 0.0
        self.agri_area = 0.0

        # topology
        self.neighbors = set()

        # state
        self.injection_mw = 0.0
        self.load_mw = 0.0
        self.available_mw = 0.0
        self.unmet_mw = 0.0
        self.surplus_mw = 0.0
        self.surplus_inventory_mwh = 0.0

        # flows (directed) for mismatch analysis
        self.boundary_flows = {}


# ---------------------------
# Mesh loading + neighbors
# ---------------------------
def load_trionchain_mesh(filename="trionchain_mesh.msh"):
    mesh = meshio.read(filename)
    points = mesh.points

    if "quad" in mesh.cells_dict:
        elements = mesh.cells_dict["quad"]
    elif "triangle" in mesh.cells_dict:
        elements = mesh.cells_dict["triangle"]
    else:
        raise RuntimeError("Mesh must contain quad or triangle elements.")

    cells = {}
    for eid, elem in enumerate(elements):
        cells[eid] = TrionCell(eid, elem, points[elem])

    # Neighbors share an edge (>=2 shared nodes)
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if len(set(elements[i]).intersection(elements[j])) >= 2:
                cells[i].neighbors.add(j)
                cells[j].neighbors.add(i)

    return cells


# ---------------------------
# Deterministic drivers
# ---------------------------
def assign_drivers(cells, seed=42):
    random.seed(seed)
    for c in cells.values():
        if random.random() < 0.12:
            c.mineral_volume = random.uniform(500, 1800)
        if random.random() < 0.16:
            c.agri_area = random.uniform(10, 70)


# ---------------------------
# Injection nodes + schedule
# ---------------------------
def select_injection_nodes(cells, k=3):
    items = []
    for cid, c in cells.items():
        x = c.coords[:, 0].mean()
        y = c.coords[:, 1].mean()
        items.append((cid, x, y))

    items_sorted = sorted(items, key=lambda t: t[2])
    south = items_sorted[len(items_sorted)//20][0]
    north = items_sorted[-len(items_sorted)//20][0]
    center = min(items, key=lambda t: (t[1]-0.5)**2 + (t[2]-0.5)**2)[0]

    chosen = [south, center, north]
    return chosen[:k]


def injection_schedule(block, base_total_mw=900.0):
    return base_total_mw * (0.9 + 0.1 * math.sin(2 * math.pi * block / 30.0))


def apply_injections(cells, injection_nodes, total_mw):
    for c in cells.values():
        c.injection_mw = 0.0

    weights = []
    for nid in injection_nodes:
        x = cells[nid].coords[:,0].mean()
        y = cells[nid].coords[:,1].mean()
        w = 1.0 + 0.8 * math.exp(-((x-0.5)**2 + (y-0.5)**2)/0.08)
        weights.append(w)

    s = sum(weights) if sum(weights) > 0 else 1.0
    for nid, w in zip(injection_nodes, weights):
        cells[nid].injection_mw = total_mw * (w / s)


# ---------------------------
# Demand model
# ---------------------------
def compute_loads(cells, block, seed=123):
    random.seed(seed)
    max_minerals = max(c.mineral_volume for c in cells.values()) or 1.0
    max_agri = max(c.agri_area for c in cells.values()) or 1.0

    for c in cells.values():
        y = c.coords[:,1].mean()
        base = 1.5
        mining = 4.0 * (c.mineral_volume / max_minerals)
        agri = 2.5 * (c.agri_area / max_agri)
        seasonal = 0.8 + 0.2 * math.sin(2 * math.pi * (block/40.0) + 3.0*y)
        c.load_mw = (base + mining + agri) * seasonal


# ============================================================
# LINE LIMITS: per-edge capacity (symmetric), per-step usage
# ============================================================

def edge_key(a, b):
    return (a, b) if a < b else (b, a)

def build_edge_capacities(cells, base_cap_mw=35.0, jitter=0.15, seed=7):
    """
    Build symmetric capacities for each undirected neighbor edge.
    Deterministic: capacity = base_cap * (1 + jitter*sin(hashlike))
    """
    random.seed(seed)
    caps = {}
    for a, ca in cells.items():
        for b in ca.neighbors:
            k = edge_key(a, b)
            if k in caps:
                continue
            # deterministic-ish variation (no true randomness needed)
            val = base_cap_mw * (1.0 + jitter * math.sin((k[0] + 1) * 12.345 + (k[1] + 1) * 7.89))
            caps[k] = max(1.0, val)
    return caps

def reset_edge_usage(edge_caps):
    return {k: 0.0 for k in edge_caps.keys()}

def edge_remaining(edge_caps, edge_usage, a, b):
    k = edge_key(a, b)
    cap = edge_caps.get(k, 0.0)
    used = edge_usage.get(k, 0.0)
    return max(cap - used, 0.0)

def add_edge_usage(edge_usage, a, b, amount):
    k = edge_key(a, b)
    edge_usage[k] = edge_usage.get(k, 0.0) + amount

def edge_utilization(edge_caps, edge_usage, a, b):
    k = edge_key(a, b)
    cap = edge_caps.get(k, 0.0)
    used = edge_usage.get(k, 0.0)
    return used / (cap + EPS)


# ---------------------------
# Flows with LINE LIMITS
# ---------------------------
def neighbor_flow_step_with_limits(cells, edge_caps, edge_usage, alpha=0.20):
    """
    Same flow step but each edge (i<->j) has capacity per step.
    We consume capacity as we send flow; if saturated, flow is clipped.
    """
    for c in cells.values():
        c.boundary_flows = {}

    delta = {cid: 0.0 for cid in cells.keys()}

    for c in cells.values():
        c.available_mw = c.injection_mw

    # deterministic iteration
    for cid in sorted(cells.keys()):
        c = cells[cid]
        net_c = c.available_mw - c.load_mw
        if net_c <= 0:
            continue

        deficit_neighbors = []
        total_def = 0.0
        for nid in sorted(c.neighbors):
            n = cells[nid]
            net_n = n.available_mw - n.load_mw
            if net_n < 0:
                d = -net_n
                deficit_neighbors.append((nid, d))
                total_def += d

        if total_def <= 0:
            continue

        send_budget = alpha * net_c

        for nid, d in deficit_neighbors:
            if send_budget <= 0:
                break

            want = send_budget * (d / total_def)
            if want <= 0:
                continue

            rem = edge_remaining(edge_caps, edge_usage, cid, nid)
            flow = min(want, rem)

            if flow <= 0:
                continue

            c.boundary_flows[nid] = c.boundary_flows.get(nid, 0.0) + flow
            delta[cid] -= flow
            delta[nid] += flow
            add_edge_usage(edge_usage, cid, nid, flow)

    for cid, d in delta.items():
        cells[cid].available_mw += d


def boundary_mismatch(cells):
    m = 0.0
    for cid, c in cells.items():
        for nid, f in c.boundary_flows.items():
            rev = cells[nid].boundary_flows.get(cid, 0.0)
            m += abs(f - rev)
    return m


# ---------------------------
# 2-hop helpers + routing
# ---------------------------
def two_hop_neighbors(cells, cid):
    one = set(cells[cid].neighbors)
    two = set()
    for n in one:
        two |= set(cells[n].neighbors)
    two.discard(cid)
    return one | two

def find_best_path_1_or_2_hop(cells, src, dst, edge_caps, edge_usage):
    """
    Returns:
      path_nodes: [src, dst] or [src, mid, dst] or None
      path_remaining: min remaining capacity along edges
      path_max_util: max utilization along edges (after usage so far)
    Choose the path with highest remaining capacity; tie-break by lower max_util then by mid id.
    """
    if dst in cells[src].neighbors:
        rem = edge_remaining(edge_caps, edge_usage, src, dst)
        maxu = edge_utilization(edge_caps, edge_usage, src, dst)
        return [src, dst], rem, maxu

    # 2-hop candidates
    best = None
    for mid in sorted(cells[src].neighbors):
        if dst not in cells[mid].neighbors:
            continue
        rem1 = edge_remaining(edge_caps, edge_usage, src, mid)
        rem2 = edge_remaining(edge_caps, edge_usage, mid, dst)
        rem = min(rem1, rem2)
        maxu = max(edge_utilization(edge_caps, edge_usage, src, mid),
                   edge_utilization(edge_caps, edge_usage, mid, dst))
        cand = ([src, mid, dst], rem, maxu)
        if best is None:
            best = cand
        else:
            # highest rem, then lower maxu
            if cand[1] > best[1] + 1e-9:
                best = cand
            elif abs(cand[1] - best[1]) <= 1e-9 and cand[2] < best[2] - 1e-9:
                best = cand
    return best if best is not None else (None, 0.0, 1.0)


def consume_path_capacity(edge_usage, path_nodes, amount):
    for i in range(len(path_nodes) - 1):
        add_edge_usage(edge_usage, path_nodes[i], path_nodes[i+1], amount)


# ---------------------------
# Market clearing with LINE LIMITS + congestion pricing
# ---------------------------
def internal_market_clearing_dynamic_2hop_with_limits(
    cells,
    edge_caps,
    edge_usage,
    base_price=40.0,
    k_local=120.0,
    beta_global=0.6,
    hop_penalty=0.08,
    congestion_k=2.0,    # extra multiplier when edges are saturated
    price_floor=5.0,
    price_cap=500.0,
):
    """
    Trade is limited by path capacity (1-hop or best 2-hop).
    Price includes congestion multiplier based on max utilization on the route:
        price *= (1 + congestion_k * max_util^2)
    """

    # local serving
    for c in cells.values():
        served = min(c.available_mw, c.load_mw)
        remaining = c.available_mw - served
        deficit = c.load_mw - served
        c.unmet_mw = deficit
        c.surplus_mw = max(remaining, 0.0)

    total_load = sum(c.load_mw for c in cells.values())
    total_unmet = sum(c.unmet_mw for c in cells.values())
    global_scarcity = total_unmet / (total_load + EPS)

    # local supply around buyer (1-hop) for stress calc
    local_supply_1hop = {}
    for bid, b in cells.items():
        s = 0.0
        for nid in b.neighbors:
            s += cells[nid].surplus_mw
        local_supply_1hop[bid] = s

    sellers = sorted([c.id for c in cells.values() if c.surplus_mw > 1e-9])
    buyers_set = set([c.id for c in cells.values() if c.unmet_mw > 1e-9])

    total_traded = 0.0
    trade_values = []
    trade_prices = []

    for sid in sellers:
        seller = cells[sid]
        if seller.surplus_mw <= 0:
            continue

        candidates = [bid for bid in two_hop_neighbors(cells, sid)
                      if bid in buyers_set and cells[bid].unmet_mw > 0]
        if not candidates:
            seller.surplus_inventory_mwh += seller.surplus_mw
            seller.surplus_mw = 0.0
            continue

        remaining_sell = seller.surplus_mw
        candidates = sorted(candidates)

        for bid in candidates:
            if remaining_sell <= 0:
                break

            buyer = cells[bid]
            if buyer.unmet_mw <= 0:
                continue

            # Find best available path (capacity-aware)
            path_nodes, path_rem, path_maxu = find_best_path_1_or_2_hop(
                cells, sid, bid, edge_caps, edge_usage
            )
            if not path_nodes or path_rem <= 0:
                continue

            # local stress
            supply_near = local_supply_1hop.get(bid, 0.0)
            local_stress = buyer.unmet_mw / (buyer.unmet_mw + supply_near + EPS)

            # distance multiplier (1-hop vs 2-hop)
            hops = len(path_nodes) - 1
            dist_mult = 1.0 + hop_penalty * max(hops - 1, 0)

            # base dynamic price
            price = base_price * (1.0 + beta_global * global_scarcity) + k_local * local_stress
            price *= dist_mult

            # congestion pricing multiplier (nonlinear)
            price *= (1.0 + congestion_k * (path_maxu ** 2))

            price = max(price_floor, min(price, price_cap))

            # capacity-limited trade
            take = min(remaining_sell, buyer.unmet_mw, path_rem)
            if take <= 0:
                continue

            buyer.unmet_mw -= take
            remaining_sell -= take

            # consume capacity on route
            consume_path_capacity(edge_usage, path_nodes, take)

            total_traded += take
            trade_values.append(take * price)
            trade_prices.append(price)

        if remaining_sell > 0:
            seller.surplus_inventory_mwh += remaining_sell
        seller.surplus_mw = 0.0

    avg_price = (sum(trade_values) / total_traded) if total_traded > 0 else 0.0
    total_unmet_after = sum(c.unmet_mw for c in cells.values())
    total_inventory = sum(c.surplus_inventory_mwh for c in cells.values())

    p10 = 0.0
    p90 = 0.0
    if trade_prices:
        srt = sorted(trade_prices)
        def q(p):
            idx = int(round(p * (len(srt) - 1)))
            return srt[idx]
        p10 = q(0.10)
        p90 = q(0.90)

    # network congestion metric: average utilization across all edges
    avg_util = 0.0
    if edge_caps:
        avg_util = sum(edge_usage[k] / (edge_caps[k] + EPS) for k in edge_caps) / len(edge_caps)

    return total_traded, avg_price, p10, p90, total_unmet_after, total_inventory, global_scarcity, avg_util


# ---------------------------
# Simulation runner
# ---------------------------
def run_simulation(
    mesh_file="trionchain_mesh.msh",
    steps=60,
    injection_nodes_k=3,
    base_total_mw=900.0,
    alpha_flow=0.20,
    # line limits
    base_cap_mw=35.0,
    cap_jitter=0.15,
    # pricing knobs
    base_price=40.0,
    k_local=120.0,
    beta_global=0.6,
    hop_penalty=0.08,
    congestion_k=2.0,
    verbose=True,
):
    cells = load_trionchain_mesh(mesh_file)
    assign_drivers(cells, seed=42)
    injection_nodes = select_injection_nodes(cells, k=injection_nodes_k)

    edge_caps = build_edge_capacities(cells, base_cap_mw=base_cap_mw, jitter=cap_jitter, seed=7)

    series = {
        "mismatch": [],
        "injected_mw": [],
        "total_load_mw": [],
        "total_traded_mw": [],
        "avg_price": [],
        "p10_price": [],
        "p90_price": [],
        "global_scarcity": [],
        "avg_edge_util": [],
        "total_unmet_mw": [],
        "total_inventory_mwh": [],
        "params": {
            "mesh_file": mesh_file,
            "steps": steps,
            "injection_nodes": injection_nodes,
            "base_total_mw": base_total_mw,
            "alpha_flow": alpha_flow,
            "line_limits": {
                "base_cap_mw": base_cap_mw,
                "cap_jitter": cap_jitter
            },
            "clearing": "2-hop + dynamic pricing + congestion + line limits",
            "base_price": base_price,
            "k_local": k_local,
            "beta_global": beta_global,
            "hop_penalty": hop_penalty,
            "congestion_k": congestion_k,
        }
    }

    for t in range(steps):
        # reset per-step capacity usage (each block has fresh thermal limits)
        edge_usage = reset_edge_usage(edge_caps)

        compute_loads(cells, t)

        total_in = injection_schedule(t, base_total_mw=base_total_mw)
        apply_injections(cells, injection_nodes, total_in)

        neighbor_flow_step_with_limits(cells, edge_caps, edge_usage, alpha=alpha_flow)
        mm = boundary_mismatch(cells)

        traded, avg_p, p10, p90, unmet, inventory, gscar, avg_util = internal_market_clearing_dynamic_2hop_with_limits(
            cells,
            edge_caps=edge_caps,
            edge_usage=edge_usage,   # NOTE: market consumes remaining capacity after flows
            base_price=base_price,
            k_local=k_local,
            beta_global=beta_global,
            hop_penalty=hop_penalty,
            congestion_k=congestion_k,
        )

        total_load = sum(c.load_mw for c in cells.values())

        series["mismatch"].append(mm)
        series["injected_mw"].append(total_in)
        series["total_load_mw"].append(total_load)
        series["total_traded_mw"].append(traded)
        series["avg_price"].append(avg_p)
        series["p10_price"].append(p10)
        series["p90_price"].append(p90)
        series["global_scarcity"].append(gscar)
        series["avg_edge_util"].append(avg_util)
        series["total_unmet_mw"].append(unmet)
        series["total_inventory_mwh"].append(inventory)

        if verbose:
            print(
                f"Step {t:02d} | inj={total_in:7.1f} | load={total_load:7.1f} | "
                f"traded={traded:6.1f} | unmet={unmet:6.1f} | inv={inventory:9.1f} | "
                f"scar={gscar:5.3f} | util={avg_util:5.3f} | "
                f"price(avg/p10/p90)={avg_p:6.1f}/{p10:5.1f}/{p90:5.1f} | mismatch={mm:7.2f}"
            )

    return series


# ---------------------------
# Plotting
# ---------------------------
def plot_results(series):
    steps = range(len(series["mismatch"]))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["mismatch"], marker="o")
    plt.title("Boundary Mismatch (Flows) – LINE LIMITS + Market")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["injected_mw"], label="Injected MW")
    plt.plot(steps, series["total_load_mw"], label="Total Load MW")
    plt.title("Power Balance (Injected vs Demand)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["total_traded_mw"], label="Traded MW")
    plt.plot(steps, series["total_unmet_mw"], label="Unmet MW")
    plt.title("Internal Market Clearing (2-hop, Line Limits)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["avg_price"], label="Avg Price")
    plt.plot(steps, series["p10_price"], label="P10 Price")
    plt.plot(steps, series["p90_price"], label="P90 Price")
    plt.title("Prices (Dynamic + Congestion)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["avg_edge_util"], label="Avg Edge Utilization")
    plt.title("Network Congestion (avg edge utilization)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Utilization (0..1+)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["total_inventory_mwh"], label="Surplus Inventory (MWh)")
    plt.plot(steps, series["global_scarcity"], label="Global Scarcity (unmet/load)")
    plt.title("Inventory & Global Scarcity")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


def save_json(series, filename="trionchain_line_limits_results.json"):
    with open(filename, "w") as f:
        json.dump(series, f, indent=2)
    print(f"Saved results to {filename}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("\nTrionChain v0.8 – LINE LIMITS + Congestion Pricing (2-hop)\n")

    mesh_file = input("Mesh file [trionchain_mesh.msh]: ").strip() or "trionchain_mesh.msh"
    steps = int(input("Number of steps [60]: ") or "60")

    # Quick knobs (optional)
    base_cap = input("Base line capacity per edge (MW) [35]: ").strip()
    base_cap_mw = float(base_cap) if base_cap else 35.0

    series = run_simulation(mesh_file=mesh_file, steps=steps, base_cap_mw=base_cap_mw, verbose=True)
    plot_results(series)

    if input("Save results to JSON? [y/N]: ").strip().lower() == "y":
        save_json(series)
