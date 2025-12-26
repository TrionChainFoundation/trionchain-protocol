# 🏗️ TrionChain Protocol

**Physics-Compliant Layer-1 Infrastructure for Real-World Assets.**

TrionChain is a sovereign blockchain built on **Substrate** that integrates **Finite Element Method (FEM)** logic into the consensus layer. It validates the physical state (Stress, Load, Generation) of critical infrastructure before settlement.

---

## 🧩 Architecture

The system operates on a dual-layer architecture designed for industrial scalability:

1.  **On-Chain (Rust/Substrate):**
    *   **TrionCell Ledger:** A specialized storage map that records the immutable physical state of geographically defined cells.
    *   **FEM Consensus:** Transaction validation logic that rejects physically impossible states (e.g., stress > limits).
    *   **Proof of Authority:** Sensors must be cryptographically authorized to write data.

2.  **Off-Chain (Python/Oracle):**
    *   **Trion Gateway:** An edge-computing node that performs sensor fusion and multiparametric FEM calculations.
    *   **Oracle Bridge:** Signs and submits verified state vectors to the blockchain via secure WebSockets.

---

## 📂 Repository Structure

*   **`/node`**: The chain configuration and P2P networking logic.
*   **`/pallets`**: Rust modules containing the FEM consensus logic (`lib.rs`).
*   **`/oracle`**: Python scripts for the **Live IoT Gateway** that connects to the blockchain.
*   **`/simulations`**: Research scripts validating the FEM math and PhyFi economic models (Heatmaps generator).

---

## 🚀 Quick Start

### Prerequisites
*   Rust & Cargo (Nightly/Stable toolchains)
*   Python 3.10+

### 1. Build the Node
```bash
cargo build --release