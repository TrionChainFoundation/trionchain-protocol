import numpy as np
import matplotlib.pyplot as plt
import meshio
import json
import random

# ============================================================
#  TRIONCHAIN SIMULATOR v0.4
#  FEM Mesh + Deterministic Consensus + Regional Shock Test
# ============================================================


# ------------------------------------------------------------
#  TrionCell = one FEM element
# ------------------------------------------------------------
class TrionCell:
    def __init__(self, eid, node_indices, coords):
        self.id = eid
        self.node_indices = node_indices
        self.coords = coords

        # RWA state
        self.energy_capacity = 0.0
        self.energy_production = 0.0
        self.mineral_volume = 0.0
        self.water_volume = 0.0
        self.agri_area = 0.0
        self.agri_yield = 0.0

        # FEM connectivity
        self.neighbors = set()
        self.boundary_flows = {}


# ------------------------------------------------------------
#  Load FEM mesh
# ------------------------------------------------------------
def load_trionchain_mesh(filename="trionchain_mesh.msh"):
    mesh = meshio.read(filename)
    points = mesh.points

    if "quad" in mesh.cells_dict:
        elements = mesh.cells_dict["quad"]
    elif "triangle" in mesh.cells_dict:
        elements = mesh.cells_dict["triangle"]
    else:
        raise RuntimeError("Mesh must contain quad or triangle elements")

    cells = {}
    for eid, elem in enumerate(elements):
        cells[eid] = TrionCell(eid, elem, points[elem])

    # FEM connectivity (shared edge)
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if len(set(elements[i]).intersection(elements[j])) >= 2:
                cells[i].neighbors.add(j)
                cells[j].neighbors.add(i)

    return cells


# ------------------------------------------------------------
#  Assign RWAs
# ------------------------------------------------------------
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


# ------------------------------------------------------------
#  Update energy & agriculture
# ------------------------------------------------------------
def update_energy(cells):
    max_mineral = max(c.mineral_volume for c in cells.values()) or 1.0
    max_water = max(c.water_volume for c in cells.values()) or 1.0

    for c in cells.values():
        if c.energy_capacity > 0:
            mf = 0.3 + 0.7 * (c.mineral_volume / max_mineral)
            wf = 0.4 + 0.6 * (c.water_volume / max_water)
            c.energy_production = c.energy_capacity * (mf + wf) / 2.0
        else:
            c.energy_production = 0.0

        if c.agri_area > 0:
            c.agri_yield = c.agri_area * (0.3 + 0.7 * (c.water_volume / max_water))
        else:
            c.agri_yield = 0.0


# ------------------------------------------------------------
#  One TrionVM-like step
# ------------------------------------------------------------
def trionvm_step(cells, alpha=0.15, depletion=0.03, water_use=0.01):
    update_energy(cells)

    for c in cells.values():
        c.boundary_flows = {}

    delta = {cid: 0.0 for cid in cells}

    for cid, c in cells.items():
        for nid in c.neighbors:
            n = cells[nid]
            diff = c.energy_production - n.energy_production
            if diff > 0:
                flow = alpha * diff
                c.boundary_flows[nid] = flow
                delta[cid] -= flow / 2
                delta[nid] += flow / 2

    for cid, d in delta.items():
        cells[cid].energy_production += d

    for c in cells.values():
        if c.energy_production > 0:
            c.mineral_volume = max(c.mineral_volume - depletion * c.energy_production, 0)
            c.water_volume = max(
                c.water_volume - water_use * (c.energy_production + c.agri_yield),
                0,
            )


# ------------------------------------------------------------
#  Boundary mismatch
# ------------------------------------------------------------
def boundary_mismatch(cells):
    m = 0.0
    for cid, c in cells.items():
        for nid, f in c.boundary_flows.items():
            m += abs(f - cells[nid].boundary_flows.get(cid, 0.0))
    return m


# ------------------------------------------------------------
#  Run simulation WITH SHOCK
# ------------------------------------------------------------
def run_simulation(mesh_file, steps=40, verbose=True):
    cells = load_trionchain_mesh(mesh_file)
    assign_rwas(cells)

    data = {
        "mismatch": [],
        "energy": [],
        "minerals": [],
        "water": [],
        "agriculture": [],
    }

    SHOCK_STEP = 10
    SHOCK_FACTOR = 0.15  # 85% water loss

    for s in range(steps):

        # -------- SHOCK EVENT --------
        if s == SHOCK_STEP:
            print("\n⚠️  SHOCK EVENT: Regional water collapse (north)\n")
            for c in cells.values():
                y_center = c.coords[:, 1].mean()
                if y_center > 0.65:
                    c.water_volume *= SHOCK_FACTOR

        trionvm_step(cells)

        data["mismatch"].append(boundary_mismatch(cells))
        data["energy"].append(sum(c.energy_production for c in cells.values()))
        data["minerals"].append(sum(c.mineral_volume for c in cells.values()))
        data["water"].append(sum(c.water_volume for c in cells.values()))
        data["agriculture"].append(sum(c.agri_yield for c in cells.values()))

        if verbose:
            print(
                f"Step {s}: "
                f"mismatch={data['mismatch'][-1]:.3f} | "
                f"energy={data['energy'][-1]:.1f} | "
                f"water={data['water'][-1]:.1f}"
            )

    return data


# ------------------------------------------------------------
#  Plot results (macOS safe)
# ------------------------------------------------------------
def plot_results(data):
    steps = range(len(data["mismatch"]))

    # Plot 1: Boundary mismatch
    plt.figure(figsize=(7, 4))
    plt.plot(steps, data["mismatch"], marker="o")
    plt.title("Boundary Mismatch – Regional Shock Test")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # Plot 2: RWA dynamics
    plt.figure(figsize=(7, 4))
    plt.plot(steps, data["energy"], label="Energy")
    plt.plot(steps, data["minerals"], label="Minerals")
    plt.plot(steps, data["water"], label="Water")
    plt.plot(steps, data["agriculture"], label="Agriculture")
    plt.title("RWA Dynamics under Regional Shock")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


# ------------------------------------------------------------
#  Save results
# ------------------------------------------------------------
def save_json(data, filename="trionchain_shock_results.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {filename}")


# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\nTrionChain FEM Shock Test Simulator v0.4\n")

    mesh_file = input("Mesh file [trionchain_mesh.msh]: ").strip() or "trionchain_mesh.msh"
    steps = int(input("Number of steps [40]: ") or "40")

    results = run_simulation(mesh_file, steps)
    plot_results(results)

    if input("Save results to JSON? [y/N]: ").lower() == "y":
        save_json(results)
