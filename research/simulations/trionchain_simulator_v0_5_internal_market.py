import math
import random
import json
import meshio
import matplotlib.pyplot as plt

# ============================================================
#  TRIONCHAIN SIMULATOR v0.5
#  FEM Mesh + Internal Energy Market (Surplus sold to deficits)
# ============================================================

class TrionCell:
    def __init__(self, eid, node_indices, coords):
        self.id = eid
        self.node_indices = node_indices
        self.coords = coords

        # RWA "capacity / resources" (optional drivers for demand/supply)
        self.energy_capacity = 0.0
        self.mineral_volume = 0.0
        self.water_volume = 0.0
        self.agri_area = 0.0

        # Network topology
        self.neighbors = set()

        # Market + power state (per step)
        self.injection_mw = 0.0     # external -> into chain (inside the sim)
        self.load_mw = 0.0          # regional demand
        self.available_mw = 0.0     # injection + received from neighbors
        self.unmet_mw = 0.0         # demand not satisfied
        self.surplus_mw = 0.0       # leftover after serving load
        self.surplus_inventory_mwh = 0.0  # accumulated surplus sold later or stored

        # For analysis
        self.boundary_flows = {}    # neighbor -> MW flow (directional)
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
# Simple RWA assignment (optional realism)
# ---------------------------
def assign_rwas(cells, seed=42):
    random.seed(seed)
    for c in cells.values():
        # energy "sites" (capacity as metadata)
        if random.random() < 0.18:
            c.energy_capacity = random.uniform(10, 80)
        if random.random() < 0.12:
            c.mineral_volume = random.uniform(500, 1800)
        if random.random() < 0.10:
            c.water_volume = random.uniform(800, 4000)
        if random.random() < 0.16:
            c.agri_area = random.uniform(10, 70)


# ---------------------------
# Pick injection nodes deterministically (no manual IDs needed)
# ---------------------------
def select_injection_nodes(cells, k=3):
    """
    Select k injection nodes by geometry:
    - one in "south" (low y)
    - one in "center"
    - one in "north" (high y)
    """
    items = []
    for cid, c in cells.items():
        x = c.coords[:, 0].mean()
        y = c.coords[:, 1].mean()
        items.append((cid, x, y))

    # Sort by y
    items_sorted = sorted(items, key=lambda t: t[2])
    south = items_sorted[len(items_sorted)//20][0]          # near bottom
    north = items_sorted[-len(items_sorted)//20][0]         # near top

    # Center = closest to (0.5, 0.5)
    center = min(items, key=lambda t: (t[1]-0.5)**2 + (t[2]-0.5)**2)[0]

    chosen = [south, center, north]

    # If k > 3, add more near center ring
    if k > 3:
        remaining = [cid for cid, _, _ in items if cid not in chosen]
        # choose additional by closeness to (0.5, 0.5)
        remaining_sorted = sorted(
            remaining,
            key=lambda cid: (cells[cid].coords[:,0].mean()-0.5)**2 + (cells[cid].coords[:,1].mean()-0.5)**2
        )
        chosen += remaining_sorted[:(k-3)]

    return chosen[:k]


# ---------------------------
# Injection schedule (MW into chain, per block)
# ---------------------------
def injection_schedule(block, base_total_mw=300.0):
    """
    Deterministic waveform: base + small sinusoidal variability.
    """
    return base_total_mw * (0.9 + 0.1 * math.sin(2 * math.pi * block / 30.0))


def apply_injections(cells, injection_nodes, block, total_mw):
    """
    Split total injected MW across injection nodes.
    """
    for c in cells.values():
        c.injection_mw = 0.0

    # weights: center a bit higher
    weights = []
    for nid in injection_nodes:
        x = cells[nid].coords[:,0].mean()
        y = cells[nid].coords[:,1].mean()
        # center-bias weight
        w = 1.0 + 0.8 * math.exp(-((x-0.5)**2 + (y-0.5)**2)/0.08)
        weights.append(w)
    s = sum(weights) if sum(weights) > 0 else 1.0

    for nid, w in zip(injection_nodes, weights):
        cells[nid].injection_mw = total_mw * (w / s)


# ---------------------------
# Demand model (regional load)
# ---------------------------
def compute_loads(cells, block, seed=123):
    """
    Deterministic per-cell demand:
    load = base + (mining + agri) + gentle seasonal term by y.
    """
    random.seed(seed)  # keep deterministic across runs

    # Normalize drivers
    max_minerals = max(c.mineral_volume for c in cells.values()) or 1.0
    max_agri = max(c.agri_area for c in cells.values()) or 1.0

    for c in cells.values():
        y = c.coords[:,1].mean()

        base = 1.5  # MW baseline
        mining = 4.0 * (c.mineral_volume / max_minerals)   # mining load
        agri = 2.5 * (c.agri_area / max_agri)              # agri load

        seasonal = 0.8 + 0.2 * math.sin(2 * math.pi * (block/40.0) + 3.0*y)
        c.load_mw = (base + mining + agri) * seasonal


# ---------------------------
# Neighbor flow step (power diffusion over FEM topology)
# ---------------------------
def neighbor_flow_step(cells, alpha=0.20):
    """
    Move MW from surplus to deficit across neighbor edges.
    This is a conservative redistribution step.
    """
    for c in cells.values():
        c.boundary_flows = {}

    # Compute net after local serving? We'll do flows on net availability.
    # Start by setting available_mw = injection + incoming will be applied via deltas.
    delta = {cid: 0.0 for cid in cells.keys()}

    # Compute current net = available - load (but available starts as injection only)
    # We'll set available now, then perform multiple passes if desired (here: single pass).
    for c in cells.values():
        c.available_mw = c.injection_mw  # start each block from injections only

    # One pass: flows based on injection imbalance (later market will finalize)
    for cid, c in cells.items():
        net_c = c.available_mw - c.load_mw
        if net_c <= 0:
            continue

        # Send to deficit neighbors proportionally
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

    # Apply deltas
    for cid, d in delta.items():
        cells[cid].available_mw += d


# ---------------------------
# Internal market clearing (sell surplus to deficits)
# ---------------------------
def internal_market_clearing(cells, base_price=50.0, k_scarcity=30.0):
    """
    After neighbor flows, perform market matching:
    - Each cell serves its own load first.
    - Surplus becomes sellable MW.
    - Deficit becomes buy demand MW.
    - Matching occurs locally with neighbors (one-hop), priced by scarcity.
    """
    # First: local balance
    for c in cells.values():
        # Serve own demand
        served = min(c.available_mw, c.load_mw)
        remaining = c.available_mw - served
        deficit = c.load_mw - served

        c.unmet_mw = deficit
        c.surplus_mw = max(remaining, 0.0)

    # Collect sellers and buyers
    sellers = [c.id for c in cells.values() if c.surplus_mw > 1e-9]
    buyers = [c.id for c in cells.values() if c.unmet_mw > 1e-9]

    total_traded = 0.0
    trade_values = []  # MW * price
    prices = []

    # For determinism: iterate in sorted order
    sellers.sort()
    buyers_set = set(buyers)

    for sid in sellers:
        seller = cells[sid]
        if seller.surplus_mw <= 0:
            continue

        # Prefer selling to neighbor deficits first
        neighbor_buyers = [nid for nid in seller.neighbors if nid in buyers_set and cells[nid].unmet_mw > 0]
        # If no neighbor buyers, keep surplus as inventory
        if not neighbor_buyers:
            seller.surplus_inventory_mwh += seller.surplus_mw  # treat MW per block as ~MWh (1 block = 1h simplification)
            seller.surplus_mw = 0.0
            continue

        # Compute scarcity among neighbor buyers
        total_def = sum(cells[bid].unmet_mw for bid in neighbor_buyers) or 1.0

        # Price = base + k * scarcity_ratio (seller sees local scarcity)
        scarcity_ratio = min(total_def / (total_def + seller.surplus_mw), 1.0)
        price = base_price + k_scarcity * scarcity_ratio
        seller.last_price = price

        remaining_sell = seller.surplus_mw

        # Allocate proportionally to deficits
        for bid in neighbor_buyers:
            if remaining_sell <= 0:
                break
            buyer = cells[bid]
            if buyer.unmet_mw <= 0:
                continue

            want = buyer.unmet_mw
            take = min(remaining_sell, want)

            # Execute trade
            buyer.unmet_mw -= take
            remaining_sell -= take
            total_traded += take
            trade_values.append(take * price)
            prices.append(price)

        # Any leftover becomes inventory
        if remaining_sell > 0:
            seller.surplus_inventory_mwh += remaining_sell
        seller.surplus_mw = 0.0

    avg_price = (sum(trade_values) / total_traded) if total_traded > 0 else 0.0
    total_unmet = sum(c.unmet_mw for c in cells.values())
    total_inventory = sum(c.surplus_inventory_mwh for c in cells.values())
    return total_traded, avg_price, total_unmet, total_inventory


# ---------------------------
# Mismatch metric (flow anti-symmetry)
# ---------------------------
def boundary_mismatch(cells):
    m = 0.0
    for cid, c in cells.items():
        for nid, f in c.boundary_flows.items():
            rev = cells[nid].boundary_flows.get(cid, 0.0)
            m += abs(f - rev)
    return m


# ---------------------------
# Simulation runner
# ---------------------------
def run_simulation(
    mesh_file="trionchain_mesh.msh",
    steps=60,
    injection_nodes_k=3,
    base_total_mw=300.0,
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
        }
    }

    for t in range(steps):
        # 1) demand
        compute_loads(cells, t)

        # 2) injections
        total_in = injection_schedule(t, base_total_mw=base_total_mw)
        apply_injections(cells, injection_nodes, t, total_in)

        # 3) neighbor redistribution (network physics)
        neighbor_flow_step(cells, alpha=alpha_flow)

        # mismatch after flow step
        mm = boundary_mismatch(cells)

        # 4) internal market (sell surplus to deficits)
        traded, avg_price, unmet, inventory = internal_market_clearing(
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
# Plotting (macOS-friendly)
# ---------------------------
def plot_results(series):
    steps = range(len(series["mismatch"]))

    # 1) mismatch
    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["mismatch"], marker="o")
    plt.title("Boundary Mismatch (Flows) – Internal Market")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # 2) injected vs load
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

    # 3) market outcomes
    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["total_traded_mw"], label="Traded MW")
    plt.plot(steps, series["total_unmet_mw"], label="Unmet MW")
    plt.title("Internal Market Clearing")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # 4) inventory + price
    plt.figure(figsize=(7, 4))
    plt.plot(steps, series["total_inventory_mwh"], label="Surplus Inventory (MWh)")
    plt.plot(steps, series["avg_price"], label="Avg Price")
    plt.title("Surplus Inventory & Price")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


def save_json(series, filename="trionchain_internal_market_results.json"):
    with open(filename, "w") as f:
        json.dump(series, f, indent=2)
    print(f"Saved results to {filename}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("\nTrionChain v0.5 – FEM Internal Energy Market Simulation\n")

    mesh_file = input("Mesh file [trionchain_mesh.msh]: ").strip() or "trionchain_mesh.msh"
    steps = int(input("Number of steps [60]: ") or "60")

    series = run_simulation(mesh_file=mesh_file, steps=steps, verbose=True)
    plot_results(series)

    if input("Save results to JSON? [y/N]: ").strip().lower() == "y":
        save_json(series)
