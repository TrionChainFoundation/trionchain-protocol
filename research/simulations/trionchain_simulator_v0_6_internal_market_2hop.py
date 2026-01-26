import math
import random
import json
import meshio
import matplotlib.pyplot as plt

# ============================================================
#  TRIONCHAIN SIMULATOR v0.6
#  FEM Mesh + Internal Energy Market (2-hop clearing)
# ============================================================

class TrionCell:
    def __init__(self, eid, node_indices, coords):
        self.id = eid
        self.node_indices = node_indices
        self.coords = coords

        # Optional demand drivers
        self.energy_capacity = 0.0
        self.mineral_volume = 0.0
        self.water_volume = 0.0
        self.agri_area = 0.0

        # Topology
        self.neighbors = set()

        # Market + power state
        self.injection_mw = 0.0
        self.load_mw = 0.0
        self.available_mw = 0.0
        self.unmet_mw = 0.0
        self.surplus_mw = 0.0
        self.surplus_inventory_mwh = 0.0

        # Analysis
        self.boundary_flows = {}
        self.last_price = 0.0


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
# Optional RWA assignment (shapes demand)
# ---------------------------
def assign_rwas(cells, seed=42):
    random.seed(seed)
    for c in cells.values():
        if random.random() < 0.18:
            c.energy_capacity = random.uniform(10, 80)
        if random.random() < 0.12:
            c.mineral_volume = random.uniform(500, 1800)
        if random.random() < 0.10:
            c.water_volume = random.uniform(800, 4000)
        if random.random() < 0.16:
            c.agri_area = random.uniform(10, 70)


# ---------------------------
# Pick injection nodes deterministically
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


# ---------------------------
# Injection schedule
# ---------------------------
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


# ---------------------------
# Neighbor redistribution (1 pass)
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
# Build 2-hop neighborhood
# ---------------------------
def two_hop_neighbors(cells, cid):
    """
    Returns set of nodes within 2 hops of cid (excluding cid).
    """
    one = set(cells[cid].neighbors)
    two = set()
    for n in one:
        two |= set(cells[n].neighbors)
    two.discard(cid)
    return one | two


# ---------------------------
# Internal market clearing (2-hop)
# ---------------------------
def internal_market_clearing_2hop(cells, base_price=50.0, k_scarcity=30.0):
    # local serving
    for c in cells.values():
        served = min(c.available_mw, c.load_mw)
        remaining = c.available_mw - served
        deficit = c.load_mw - served
        c.unmet_mw = deficit
        c.surplus_mw = max(remaining, 0.0)

    sellers = sorted([c.id for c in cells.values() if c.surplus_mw > 1e-9])
    buyers_set = set([c.id for c in cells.values() if c.unmet_mw > 1e-9])

    total_traded = 0.0
    trade_values = []

    for sid in sellers:
        seller = cells[sid]
        if seller.surplus_mw <= 0:
            continue

        # 2-hop buyer candidates
        candidates = [bid for bid in two_hop_neighbors(cells, sid) if bid in buyers_set and cells[bid].unmet_mw > 0]

        if not candidates:
            seller.surplus_inventory_mwh += seller.surplus_mw
            seller.surplus_mw = 0.0
            continue

        total_def = sum(cells[bid].unmet_mw for bid in candidates) or 1.0
        scarcity_ratio = min(total_def / (total_def + seller.surplus_mw), 1.0)
        price = base_price + k_scarcity * scarcity_ratio
        seller.last_price = price

        remaining_sell = seller.surplus_mw

        # Allocate proportionally to deficits (stable deterministic order)
        candidates = sorted(candidates)
        for bid in candidates:
            if remaining_sell <= 0:
                break
            buyer = cells[bid]
            if buyer.unmet_mw <= 0:
                continue

            take = min(remaining_sell, buyer.unmet_mw)
            buyer.unmet_mw -= take
            remaining_sell -= take
            total_traded += take
            trade_values.append(take * price)

        if remaining_sell > 0:
            seller.surplus_inventory_mwh += remaining_sell
        seller.surplus_mw = 0.0

    avg_price = (sum(trade_values) / total_traded) if total_traded > 0 else 0.0
    total_unmet = sum(c.unmet_mw for c in cells.values())
    total_inventory = sum(c.surplus_inventory_mwh for c in cells.values())
    return total_traded, avg_price, total_unmet, total_inventory


# ---------------------------
# Simulation runner
# ---------------------------
def run_simulation(
    mesh_file="trionchain_mesh.msh",
    steps=60,
    injection_nodes_k=3,
    base_total_mw=900.0,
    alpha_flow=0.20,
    base_price=50.0,
    k_scarcity=30.0,
    seed_rwa=42,
    verbose=True,
):
    cells = load_trionchain_mesh(mesh_file)
    assign_rwas(cells, seed=seed_rwa)
    injection_nodes = select_injection_nodes(cells, k=injection_nodes_k)

    series = {
        "mismatch": [],
        "injected_mw": [],
        "total_load_mw": [],
        "total_traded_mw": [],
        "avg_price": [],
        "total_unmet_mw": [],
        "total_inventory_mwh": [],
        "params": {
            "mesh_file": mesh_file,
            "steps": steps,
            "injection_nodes": injection_nodes,
            "base_total_mw": base_total_mw,
            "alpha_flow": alpha_flow,
            "base_price": base_price,
            "k_scarcity": k_scarcity,
            "clearing": "2-hop",
        }
    }

    for t in range(steps):
        compute_loads(cells, t)

        total_in = injection_schedule(t, base_total_mw=base_total_mw)
        apply_injections(cells, injection_nodes, total_in)

        neighbor_flow_step(cells, alpha=alpha_flow)
        mm = boundary_mismatch(cells)

        traded, avg_price, unmet, inventory = internal_market_clearing_2hop(
            cells, base_price=base_price, k_scarcity=k_scarcity
        )

        total_load = sum(c.load_mw for c in cells.values())

        series["mismatch"].append(mm)
        series["injected_mw"].append(total_in)
        series["total_load_mw"].append(total_load)
        series["total_traded_mw"].append(traded)
        series["avg_price"].append(avg_price)
        series["total_unmet_mw"].append(unmet)
        series["total_inventory_mwh"].append(inventory)

        if verbose:
            print(
                f"Step {t:02d} | inj={total_in:7.1f} MW | load={total_load:7.1f} MW | "
                f"traded={traded:6.1f} MW | unmet={unmet:6.1f} MW | "
                f"inv={inventory:8.1f} MWh | price={avg_price:6.1f} | mismatch={mm:7.2f}"
            )

    return series


# ---------------------------
# Plotting
# ---------------------------
def plot_results(series):
    steps = range(len(series["mismatch"]))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["mismatch"], marker="o")
    plt.title("Boundary Mismatch (Flows) – Internal Market (2-hop)")
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
    plt.title("Internal Market Clearing (2-hop)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["total_inventory_mwh"], label="Surplus Inventory (MWh)")
    plt.plot(steps, series["avg_price"], label="Avg Price")
    plt.title("Surplus Inventory & Price (2-hop)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


def save_json(series, filename="trionchain_internal_market_2hop_results.json"):
    with open(filename, "w") as f:
        json.dump(series, f, indent=2)
    print(f"Saved results to {filename}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("\nTrionChain v0.6 – FEM Internal Energy Market (2-hop clearing)\n")

    mesh_file = input("Mesh file [trionchain_mesh.msh]: ").strip() or "trionchain_mesh.msh"
    steps = int(input("Number of steps [60]: ") or "60")

    series = run_simulation(mesh_file=mesh_file, steps=steps, verbose=True)
    plot_results(series)

    if input("Save results to JSON? [y/N]: ").strip().lower() == "y":
        save_json(series)
