# 🏗️ TrionChain Protocol

**A physics-aware Layer-1 blockchain for real-world infrastructure and Physical RWA.**

TrionChain is a sovereign Substrate-based blockchain designed to **validate physical reality at the consensus layer**.  
Instead of trusting raw oracle data, the chain enforces **physics-inspired constraints** (mesh, neighborhood bounds, limits) before accepting state updates.

This repository contains a **running Layer-1**, an **active DevNet**, and a **reproducible demo**.

---

## ✅ Current Status (Grant-Relevant)

✔️ Substrate-based sovereign Layer-1  
✔️ Custom runtime pallets integrated  
✔️ Mesh / cell-based state model  
✔️ Physical consistency checks at transaction validation  
✔️ **Live DevNet running on a public server**  
✔️ Verifiable via Polkadot.js Apps  

> This is not a prototype. It is a running blockchain with novel validation logic.

---

## 🌐 Live DevNet — Verify in 30 Seconds

**WebSocket RPC**
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
