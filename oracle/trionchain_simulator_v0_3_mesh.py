import numpy as np
import matplotlib.pyplot as plt
import meshio
import json
import random

# ============================================================
#  TRIONCHAIN SIMULATOR v0.3 (FEM MESH)
#  - Uses a real FEM mesh from Gmsh (trionchain_mesh.msh)
#  - Each mesh element is a TrionCell
#  - Neighbors are defined by FEM connectivity (shared edges)
#  - Dynamic RWA: energy, minerals, water, agriculture
# ============================================================


# ------------------------------------------------------------
#  TrionObject: RWA abstraction (optional for extensions)
# ------------------------------------------------------------
class TrionObject:
    """
    Basic RWA representation in a specific region (TrionCell).
    obj_type can be:
      - "energy_plant"
      - "mineral_reservoir"
      - "water_basin"
      - "agriculture_field"
      - "land_parcel"
    """
    def __init__(self, object_id, obj_type, region_id, physical_value, economic_value):
        self.object_id = object_id
        self.obj_type = obj_type
        self.region_id = region_id  # matches TrionCell.id
        self.physical_value = physical_value
        self.economic_value = economic_value


# ------------------------------------------------------------
#  TrionCell: one FEM element in the mesh
# ------------------------------------------------------------
class TrionCell:
    """
    A TrionCell corresponds to one FEM element.
    It holds:
      - element ID
      - node indices
      - coordinates of its nodes
      - local RWA-related state
      - neighbors (other TrionCells that share an edge)
    """
    def __init__(self, eid, node_indices, coords):
        self.id = eid
        self.node_indices = node_indices  # indices into mesh.points
        self.coords = coords              # array of shape (num_nodes, 3)

        # Local state (dynamic)
        self.energy_capacity = 0.0
        self.energy_production = 0.0
        self.mineral_volume = 0.0
        self.water_volume = 0.0
        self.agri_area = 0.0
        self.agri_yield = 0.0

        # Neighbors (set of cell IDs)
        self.neighbors = set()

        # Boundary flows (to neighbors)
        self.boundary_flows = {}

        # Optional list of TrionObjects
        self.objects = []


# ------------------------------------------------------------
#  Load FEM mesh and build TrionCells + connectivity
# ------------------------------------------------------------
def load_trionchain_mesh(filename="trionchain_mesh.msh"):
    """
    Load a Gmsh mesh and convert it into:
      - dict of TrionCell objects: {eid: TrionCell}
      - FEM-based neighbor connectivity
    """
    mesh = meshio.read(filename)

    points = mesh.points  # Nx3 array of node coordinates

    # Prefer quadrilateral elements; fallback to triangles
    if "quad" in mesh.cells_dict:
        elements = mesh.cells_dict["quad"]
    elif "triangle" in mesh.cells_dict:
        elements = mesh.cells_dict["triangle"]
    else:
        raise ValueError("Mesh must contain 'quad' or 'triangle' elements.")

    # Create TrionCells
    cells = {}
    for eid, elem in enumerate(elements):
        node_indices = elem
        coords = points[node_indices]
        cells[eid] = TrionCell(eid, node_indices, coords)

    # Build connectivity: neighbors share at least 2 nodes (an edge in 2D)
    num_elems = len(elements)
    for i in range(num_elems):
        for j in range(i + 1, num_elems):
            shared_nodes = set(elements[i]).intersection(set(elements[j]))
            if len(shared_nodes) >= 2:
                cells[i].neighbors.add(j)
                cells[j].neighbors.add(i)

    return cells


# ------------------------------------------------------------
#  Initialize RWAs (RWA = TrionObjects / local state)
# ------------------------------------------------------------
def assign_random_rwas(cells, seed=42):
    """
    Assign random RWA-like values to each TrionCell.
    This is a simple heuristic demo; later this can be replaced
    by geographic / domain-specific logic.
    """
    random.seed(seed)

    for cell in cells.values():
        # Probability of each RWA type appearing in this cell
        if random.random() < 0.18:
            # Energy plant
            capacity = random.uniform(10, 80)  # MW
            cell.energy_capacity += capacity

        if random.random() < 0.12:
            # Mineral reservoir
            volume = random.uniform(500, 1800)
            cell.mineral_volume += volume

        if random.random() < 0.10:
            # Water basin
            volume = random.uniform(800, 4000)
            cell.water_volume += volume

        if random.random() < 0.16:
            # Agriculture field
            area = random.uniform(10, 70)  # hectares
            cell.agri_area += area


# ------------------------------------------------------------
#  Update energy & agriculture yield from RWA state
# ------------------------------------------------------------
def update_energy_from_rwa(cells):
    """
    Update energy production and agriculture yield based on:
      - energy capacity
      - mineral availability
      - water availability
    """
    # Max values for normalization
    max_minerals = max((cell.mineral_volume for cell in cells.values()), default=1.0)
    max_water = max((cell.water_volume for cell in cells.values()), default=1.0)

    if max_minerals <= 0:
        max_minerals = 1.0
    if max_water <= 0:
        max_water = 1.0

    for cell in cells.values():
        if cell.energy_capacity > 0:
            mineral_factor = 0.3 + 0.7 * (cell.mineral_volume / max_minerals)
            water_factor = 0.4 + 0.6 * (cell.water_volume / max_water)
            combined_factor = 0.5 * mineral_factor + 0.5 * water_factor
            cell.energy_production = cell.energy_capacity * combined_factor
        else:
            cell.energy_production = 0.0

        # Simple agriculture yield model driven by water
        if cell.agri_area > 0:
            cell.agri_yield = cell.agri_area * (0.3 + 0.7 * (cell.water_volume / max_water))
        else:
            cell.agri_yield = 0.0


# ------------------------------------------------------------
#  One TrionVM-like step over the FEM mesh
# ------------------------------------------------------------
def trionvm_step(cells, alpha=0.15, depletion_rate=0.03, water_use_rate=0.01):
    """
    One simulation step:
      1) Update energy from RWA
      2) Compute boundary flows (energy diffusion) and adjust production
      3) Deplete minerals and water based on activity
    """
    # Step 1: update local production & yield
    update_energy_from_rwa(cells)

    # Clear previous flows
    for cell in cells.values():
        cell.boundary_flows = {}

    # Plan adjustments
    delta_prod = {cid: 0.0 for cid in cells.keys()}

    # Step 2: compute flows between neighbors
    for cid, cell in cells.items():
        for nid in cell.neighbors:
            neighbor = cells[nid]
            diff = cell.energy_production - neighbor.energy_production
            if diff > 0:
                flow = alpha * diff
                # record flow from cid to nid
                cell.boundary_flows[nid] = flow
                # symmetrical adjustments
                delta_prod[cid] -= flow / 2.0
                delta_prod[nid] += flow / 2.0

    # Apply energy production changes
    for cid, delta in delta_prod.items():
        cells[cid].energy_production += delta

    # Step 3: deplete minerals and water
    for cell in cells.values():
        if cell.energy_production > 0 and cell.mineral_volume > 0:
            depletion = depletion_rate * cell.energy_production
            cell.mineral_volume = max(cell.mineral_volume - depletion, 0.0)

        if (cell.energy_production > 0 or cell.agri_yield > 0) and cell.water_volume > 0:
            water_use = water_use_rate * (cell.energy_production + cell.agri_yield)
            cell.water_volume = max(cell.water_volume - water_use, 0.0)


# ------------------------------------------------------------
#  Boundary mismatch metric (FEM neighbor-based)
# ------------------------------------------------------------
def boundary_mismatch(cells):
    """
    Sum of |flow_ij - flow_ji| over all neighboring TrionCells.
    This acts as a global coherence / convergence metric.
    """
    total = 0.0
    for cid, cell in cells.items():
        for nid, flow in cell.boundary_flows.items():
            reverse_flow = cells[nid].boundary_flows.get(cid, 0.0)
            total += abs(flow - reverse_flow)
    return total


# ------------------------------------------------------------
#  Run full simulation
# ------------------------------------------------------------
def run_simulation(
    mesh_filename="trionchain_mesh.msh",
    num_steps=40,
    alpha=0.15,
    depletion_rate=0.03,
    water_use_rate=0.01,
    seed=42,
    verbose=False,
):
    """
    Run a full TrionChain FEM-based simulation.
    Returns a dict of time series.
    """
    # Load mesh and create cells from FEM
    cells = load_trionchain_mesh(mesh_filename)

    # Assign random RWA distribution
    assign_random_rwas(cells, seed=seed)

    mismatches = []
    total_energy = []
    total_minerals = []
    total_water = []
    total_agri_yield = []

    for step in range(num_steps):
        trionvm_step(
            cells,
            alpha=alpha,
            depletion_rate=depletion_rate,
            water_use_rate=water_use_rate,
        )

        mismatches.append(boundary_mismatch(cells))
        total_energy.append(sum(c.energy_production for c in cells.values()))
        total_minerals.append(sum(c.mineral_volume for c in cells.values()))
        total_water.append(sum(c.water_volume for c in cells.values()))
        total_agri_yield.append(sum(c.agri_yield for c in cells.values()))

        if verbose:
            print(
                f"Step {step}: "
                f"mismatch={mismatches[-1]:.3f}, "
                f"energy={total_energy[-1]:.2f}, "
                f"minerals={total_minerals[-1]:.2f}, "
                f"water={total_water[-1]:.2f}, "
                f"agri_yield={total_agri_yield[-1]:.2f}"
            )

    return {
        "mismatches": mismatches,
        "total_energy": total_energy,
        "total_minerals": total_minerals,
        "total_water": total_water,
        "total_agri_yield": total_agri_yield,
        "params": {
            "mesh_filename": mesh_filename,
            "num_steps": num_steps,
            "alpha": alpha,
            "depletion_rate": depletion_rate,
            "water_use_rate": water_use_rate,
            "seed": seed,
        },
    }


# ------------------------------------------------------------
#  Save + Plot
# ------------------------------------------------------------
def save_results_to_json(results, filename="trionchain_mesh_sim_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


def plot_results(results):
    steps = range(len(results["mismatches"]))

    # Plot mismatch
    plt.figure(figsize=(6, 4))
    plt.plot(steps, results["mismatches"], marker="o")
    plt.title("Boundary Mismatch (FEM Mesh, Dynamic RWA)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot RWA dynamics
    plt.figure(figsize=(6, 4))
    plt.plot(steps, results["total_energy"], marker="o", label="Total Energy")
    plt.plot(steps, results["total_minerals"], marker="s", label="Total Minerals")
    plt.plot(steps, results["total_water"], marker="^", label="Total Water")
    plt.plot(steps, results["total_agri_yield"], marker="x", label="Total Agri Yield")
    plt.title("RWA Dynamics on FEM Mesh")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Value (arbitrary units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
#  Main entry point (CLI)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("TrionChain FEM Mesh Simulator v0.3")

    mesh_file = input("Mesh filename [trionchain_mesh.msh]: ").strip() or "trionchain_mesh.msh"

    try:
        num_steps = int(input("Number of simulation steps [40]: ") or "40")
        alpha = float(input("Flow coefficient alpha [0.15]: ") or "0.15")
        depletion_rate = float(input("Mineral depletion rate [0.03]: ") or "0.03")
        water_use_rate = float(input("Water use rate [0.01]: ") or "0.01")
    except ValueError:
        print("Invalid input, using default parameters.")
        num_steps, alpha, depletion_rate, water_use_rate = 40, 0.15, 0.03, 0.01

    results = run_simulation(
        mesh_filename=mesh_file,
        num_steps=num_steps,
        alpha=alpha,
        depletion_rate=depletion_rate,
        water_use_rate=water_use_rate,
        verbose=True,
    )

    plot_results(results)

    save = input("Save results to JSON file? [y/N]: ").strip().lower()
    if save == "y":
        filename = input("Filename [trionchain_mesh_sim_results.json]: ").strip() or "trionchain_mesh_sim_results.json"
        save_results_to_json(results, filename)
