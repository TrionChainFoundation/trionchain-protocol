import math
import random
import json
import meshio
import matplotlib.pyplot as plt

# ============================================================
#  TRIONCHAIN SIMULATOR v0.7
#  FEM Mesh + Internal Energy Market (2-hop) + Dynamic Prices
# ============================================================

EPS = 1e-12

class TrionCell:
    def __init__(self, eid, node_indices, coords):
        self.id = eid
        self.node_indices = node_indices
        self.coords = coords

        # Optional demand drivers
        self.mineral_volume = 0.0
        self.agri_area = 0.0

        # Topology
        self.neighbors = set()

        # Power state
        self.injection_mw = 0.0
        self.load_mw = 0.0
        self.available_mw = 0.0
        self.unmet_mw = 0.0
        self.surplus_mw = 0.0
        self.surplus_inventory_mwh = 0.0

        # Flows analysis
        self.boundary_flows = {}


# ---------------------------
# Mesh loading
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
# Assign drivers (deterministic)
# ---------------------------
def assign_drivers(cells, seed=42):
    random.seed(seed)
    for c in cells.values():
        if random.random() < 0.12:
            c.mineral_volume = random.uniform(500, 1800)
        if random.random() < 0.16:
            c.agri_area = random.uniform(10, 70)


# ---------------------------
# Injection nodes
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

    if k > 3:
        remaining = [cid for cid, _, _ in items if cid not in chosen]
        remaining_sorted = sorted(
            remaining,
            key=lambda cid: (cells[cid].coords[:,0].mean()-0.5)**2 + (cells[cid].coords[:,1].mean()-0.5)**2
        )
        chosen += remaining_sorted[:(k-3)]

    return chosen[:k]


def injection_schedule(block, base_total_mw=900.0):
    return base_total_mw * (0.9 + 0.1 * math.sin(2 * math.pi * block / 30.0))


def apply_injections(cells, injection_nodes, total_mw):
    for c in cells.values():
        c.injection_mw = 0.0

    # center-biased weights
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


# ---------------------------
# Flows (one pass)
# ---------------------------
def neighbor_flow_step(cells, alpha=0.20):
    for c in cells.values():
        c.boundary_flows = {}

    delta = {cid: 0.0 for cid in cells.keys()}

    for c in cells.values():
        c.available_mw = c.injection_mw

    for cid, c in cells.items():
        net_c = c.available_mw - c.load_mw
        if net_c <= 0:
            continue

        deficit_neighbors = []
        total_def = 0.0
        for nid in c.neighbors:
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
            flow = send_budget * (d / total_def)
            if flow <= 0:
                continue
            c.boundary_flows[nid] = flow
            delta[cid] -= flow
            delta[nid] += flow

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
# 2-hop neighborhood helpers
# ---------------------------
def two_hop_neighbors(cells, cid):
    one = set(cells[cid].neighbors)
    two = set()
    for n in one:
        two |= set(cells[n].neighbors)
    two.discard(cid)
    return one | two


def hop_distance(cells, src, dst):
    """Return 1 if neighbor, 2 if within 2 hops, else 99."""
    if dst in cells[src].neighbors:
        return 1
    if dst in two_hop_neighbors(cells, src):
        return 2
    return 99


# ---------------------------
# Dynamic pricing market (2-hop)
# ---------------------------
def internal_market_clearing_dynamic_2hop(
    cells,
    base_price=40.0,
    k_local=120.0,
    beta_global=0.6,
    hop_penalty=0.08,
    price_floor=5.0,
    price_cap=500.0,
):
    """
    Dynamic price = base * (1 + beta_global * global_scarcity)
                   + k_local * local_stress
                   then * (1 + hop_penalty*(hops-1))

    where:
      global_scarcity ~ total_unmet / total_load
      local_stress ~ buyer_unmet / (buyer_unmet + local_supply_near_buyer)

    Result: price reacts to congestion/scarcity, but remains deterministic.
    """

    # Local serving
    for c in cells.values():
        served = min(c.available_mw, c.load_mw)
        remaining = c.available_mw - served
        deficit = c.load_mw - served
        c.unmet_mw = deficit
        c.surplus_mw = max(remaining, 0.0)

    total_load = sum(c.load_mw for c in cells.values())
    total_unmet = sum(c.unmet_mw for c in cells.values())
    global_scarcity = total_unmet / (total_load + EPS)

    # Precompute "local supply near buyer": sum of surplus in 1-hop neighborhood
    # (computed once at the start of clearing, deterministic)
    local_supply_1hop = {}
    for bid, b in cells.items():
        s = 0.0
        for nid in b.neighbors:
            s += cells[nid].surplus_mw
        local_supply_1hop[bid] = s

    sellers = sorted([c.id for c in cells.values() if c.surplus_mw > 1e-9])
    buyers_set = set([c.id for c in cells.values() if c.unmet_mw > 1e-9])

    total_traded = 0.0
    trade_values = []   # MW * price
    trade_prices = []   # price per trade (for quantiles)

    for sid in sellers:
        seller = cells[sid]
        if seller.surplus_mw <= 0:
            continue

        # Buyer candidates within 2 hops
        candidates = [bid for bid in two_hop_neighbors(cells, sid) if bid in buyers_set and cells[bid].unmet_mw > 0]
        if not candidates:
            seller.surplus_inventory_mwh += seller.surplus_mw
            seller.surplus_mw = 0.0
            continue

        remaining_sell = seller.surplus_mw

        # Deterministic order
        candidates = sorted(candidates)

        for bid in candidates:
            if remaining_sell <= 0:
                break

            buyer = cells[bid]
            if buyer.unmet_mw <= 0:
                continue

            # Local stress (higher unmet + low nearby supply => higher stress)
            supply_near_buyer = local_supply_1hop.get(bid, 0.0)
            local_stress = buyer.unmet_mw / (buyer.unmet_mw + supply_near_buyer + EPS)

            # Distance penalty
            hops = hop_distance(cells, sid, bid)
            dist_mult = 1.0 + hop_penalty * max(hops - 1, 0)

            # Dynamic price
            price = base_price * (1.0 + beta_global * global_scarcity) + k_local * local_stress
            price *= dist_mult
            price = max(price_floor, min(price, price_cap))

            take = min(remaining_sell, buyer.unmet_mw)

            buyer.unmet_mw -= take
            remaining_sell -= take

            total_traded += take
            trade_values.append(take * price)
            trade_prices.append(price)

        # leftover => inventory
        if remaining_sell > 0:
            seller.surplus_inventory_mwh += remaining_sell
        seller.surplus_mw = 0.0

    avg_price = (sum(trade_values) / total_traded) if total_traded > 0 else 0.0
    total_unmet_after = sum(c.unmet_mw for c in cells.values())
    total_inventory = sum(c.surplus_inventory_mwh for c in cells.values())

    # Quantiles (p10/p90) for price dispersion
    p10 = 0.0
    p90 = 0.0
    if trade_prices:
        trade_prices_sorted = sorted(trade_prices)
        def q(p):
            idx = int(round(p * (len(trade_prices_sorted) - 1)))
            return trade_prices_sorted[idx]
        p10 = q(0.10)
        p90 = q(0.90)

    return total_traded, avg_price, p10, p90, total_unmet_after, total_inventory, global_scarcity


# ---------------------------
# Simulation runner
# ---------------------------
def run_simulation(
    mesh_file="trionchain_mesh.msh",
    steps=60,
    injection_nodes_k=3,
    base_total_mw=900.0,
    alpha_flow=0.20,
    # pricing knobs
    base_price=40.0,
    k_local=120.0,
    beta_global=0.6,
    hop_penalty=0.08,
    verbose=True,
):
    cells = load_trionchain_mesh(mesh_file)
    assign_drivers(cells, seed=42)
    injection_nodes = select_injection_nodes(cells, k=injection_nodes_k)

    series = {
        "mismatch": [],
        "injected_mw": [],
        "total_load_mw": [],
        "total_traded_mw": [],
        "avg_price": [],
        "p10_price": [],
        "p90_price": [],
        "global_scarcity": [],
        "total_unmet_mw": [],
        "total_inventory_mwh": [],
        "params": {
            "mesh_file": mesh_file,
            "steps": steps,
            "injection_nodes": injection_nodes,
            "base_total_mw": base_total_mw,
            "alpha_flow": alpha_flow,
            "clearing": "2-hop + dynamic pricing",
            "base_price": base_price,
            "k_local": k_local,
            "beta_global": beta_global,
            "hop_penalty": hop_penalty,
        }
    }

    for t in range(steps):
        compute_loads(cells, t)

        total_in = injection_schedule(t, base_total_mw=base_total_mw)
        apply_injections(cells, injection_nodes, total_in)

        neighbor_flow_step(cells, alpha=alpha_flow)
        mm = boundary_mismatch(cells)

        traded, avg_p, p10, p90, unmet, inventory, gscar = internal_market_clearing_dynamic_2hop(
            cells,
            base_price=base_price,
            k_local=k_local,
            beta_global=beta_global,
            hop_penalty=hop_penalty,
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
        series["total_unmet_mw"].append(unmet)
        series["total_inventory_mwh"].append(inventory)

        if verbose:
            print(
                f"Step {t:02d} | inj={total_in:7.1f} | load={total_load:7.1f} | "
                f"traded={traded:6.1f} | unmet={unmet:6.1f} | inv={inventory:9.1f} | "
                f"scar={gscar:5.3f} | price(avg/p10/p90)={avg_p:6.1f}/{p10:5.1f}/{p90:5.1f} | mismatch={mm:7.2f}"
            )

    return series


# ---------------------------
# Plotting
# ---------------------------
def plot_results(series):
    steps = range(len(series["mismatch"]))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["mismatch"], marker="o")
    plt.title("Boundary Mismatch (Flows) – Dynamic Pricing (2-hop)")
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
    plt.title("Internal Market Clearing (2-hop, Dynamic Pricing)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # Price dispersion
    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["avg_price"], label="Avg Price")
    plt.plot(steps, series["p10_price"], label="P10 Price")
    plt.plot(steps, series["p90_price"], label="P90 Price")
    plt.title("Dynamic Prices (Avg / P10 / P90)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # Inventory + scarcity (separate so it reads cleanly)
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


def save_json(series, filename="trionchain_dynamic_pricing_results.json"):
    with open(filename, "w") as f:
        json.dump(series, f, indent=2)
    print(f"Saved results to {filename}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("\nTrionChain v0.7 – Dynamic Pricing Internal Market (2-hop)\n")

    mesh_file = input("Mesh file [trionchain_mesh.msh]: ").strip() or "trionchain_mesh.msh"
    steps = int(input("Number of steps [60]: ") or "60")

    series = run_simulation(mesh_file=mesh_file, steps=steps, verbose=True)
    plot_results(series)

    if input("Save results to JSON? [y/N]: ").strip().lower() == "y":
        save_json(series)
