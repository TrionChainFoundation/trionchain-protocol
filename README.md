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

## 🧩 ArchiteThe system operates on a dual-layer architecture designed for industrial scalability:

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
# 🏗️ TrionChain Protocol

**Physics-Compliant Layer-1 Blockchain for Real-World Infrastructure**

TrionChain is a Substrate-based Layer-1 blockchain designed to **validate physical reality before settlement**.  
It integrates **Finite Element Method (FEM)** logic directly into the consensus layer to prevent physically impossible state transitions.

This repository contains the **full TrionChain protocol implementation**, including the node, runtime, pallets, oracles, simulations, and documentation.

---

## 🚀 What Exists Today (Not a Concept)

✅ Running Substrate-based Layer-1  
✅ Custom runtime & pallets  
✅ Mesh-based physical state model (TrionCells)  
✅ Oracle framework (Python)  
✅ React dashboard  
✅ Live devnet producing blocks on VPS  
✅ FEM & PhyFi research validated via simulations  

---

## 🧠 Core Idea

> **If a physical state is impossible, the transaction is invalid.**

Examples:
- Stress exceeding material limits → rejected
- Energy flows violating capacity → rejected
- Infrastructure state inconsistent with FEM constraints → rejected

This enables **verifiable physical settlement** for RWAs, energy systems, transport, and industrial infrastructure.

---

## 🧩 Architecture Overview

### On-Chain (Rust / Substrate)
- Custom node & runtime
- FEM-aware pallets
- TrionCell state model
- Authority-based block production
- Deterministic execution

### Off-Chain (Oracle Layer)
- Python oracle framework
- Physical state resolution
- FEM pre-validation
- Secure submission via WebSockets

## 📂 Repository Structure

/node → Substrate node configuration
/runtime → Chain runtime
/pallets → FEM & physical-state pallets
/oracles → Active oracle implementations
/dashboard → React visualization UI
/research → FEM & PhyFi simulations
/docs → Whitepaper, technical brief, grant docs
/scripts → Environment & setup tools


---

## 📄 Grant & Technical Documentation

👉 **Grant-specific documents**
- [`docs/grant/current-state-and-roadmap.md`](docs/grant/current-state-and-roadmap.md)
- [`GRANT_PROPOSAL.md`](GRANT_PROPOSAL.md)

👉 **Technical documentation**
- Whitepaper v2.1 (PDF & MD)
- Technical Brief (PDF & MD)

---

## 🧪 Live DevNet

A TrionChain development network is currently running on a VPS and continuously producing blocks.

RPC access is available during review for technical validation.

---

## 🛠️ Build the Node (Quick Start)

```bash
cargo build --release
./target/release/solochain-template-node --dev

