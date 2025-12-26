import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# TrionChain – Dual-Layer Architecture (Slide 7)
# ==============================

plt.style.use("dark_background")

def generate_trionchain_dual_layer_slide(
    out_path="slide_07_trionchain_dual_layer.png"
):
    fig = plt.figure(figsize=(14, 9), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Remove axes and grids
    ax.set_axis_off()
    ax.grid(False)

    # Absolute black background
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # --------------------------
    # Mesh definition
    # --------------------------
    x = np.linspace(-6, 6, 22)
    y = np.linspace(-6, 6, 22)
    X, Y = np.meshgrid(x, y)

    # Vertical separation of layers
    Z_protocol = np.zeros_like(X) - 3.5
    Z_settlement = np.zeros_like(X) + 3.5

    # --------------------------
    # Draw mesh layers
    # --------------------------
    # Protocol layer (TRN) – Neon cyan
    ax.plot_wireframe(
        X, Y, Z_protocol,
        color="#00FFFF",
        alpha=0.35,
        linewidth=0.6
    )

    # Settlement layer – Neutral white
    ax.plot_wireframe(
        X, Y, Z_settlement,
        color="#FFFFFF",
        alpha=0.30,
        linewidth=0.6
    )

    # --------------------------
    # Nodes (shared spatial footprint)
    # --------------------------
    np.random.seed(42)
    num_nodes = 18
    nodes_x = np.random.uniform(-5, 5, num_nodes)
    nodes_y = np.random.uniform(-5, 5, num_nodes)

    # Validators (Protocol layer)
    ax.scatter(
        nodes_x, nodes_y, -3.5,
        c="#00FFFF",
        s=120,
        edgecolors="white",
        alpha=0.95
    )

    # Clients / Institutions (Settlement layer)
    ax.scatter(
        nodes_x, nodes_y, 3.5,
        c="#FFFFFF",
        s=120,
        edgecolors="gray",
        alpha=0.95
    )

    # --------------------------
    # Vertical interactions
    # --------------------------
    for i in range(num_nodes):
        ax.plot(
            [nodes_x[i], nodes_x[i]],
            [nodes_y[i], nodes_y[i]],
            [-3.5, 3.5],
            color="gray",
            linestyle="--",
            linewidth=0.9,
            alpha=0.5
        )

    # --------------------------
    # Text annotations
    # --------------------------
    ax.text(
        0, -8.2, 4.4,
        "SETTLEMENT LAYER\n(Fiat · Stablecoins · RWAs · Institutions)",
        color="white",
        fontsize=15,
        weight="bold",
        ha="center"
    )

    ax.text(
        0, -8.2, -4.6,
        "PROTOCOL SECURITY LAYER\n(TRN Token · Validators · Consensus)",
        color="#00FFFF",
        fontsize=15,
        weight="bold",
        ha="center"
    )

    ax.text(
        7.2, 0, 0,
        "Deterministic\nConsensus\n& Validation",
        color="gray",
        fontsize=11,
        ha="center",
        va="center"
    )

    # --------------------------
    # Camera & limits
    # --------------------------
    ax.view_init(elev=22, azim=-58)
    ax.set_zlim(-6, 6)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    # --------------------------
    # Export
    # --------------------------
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.show()

    print(f"Slide exported to: {out_path}")


if __name__ == "__main__":
    generate_trionchain_dual_layer_slide()
