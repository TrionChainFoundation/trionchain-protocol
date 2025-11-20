# TrionChain Architecture

This document describes the **high-level architecture** of the TrionChain FEM-based blockchain.

## 1. Design Principles

1. **FEM-inspired mesh**
   - The world is partitioned into **cells** (similar to finite elements).
   - Each cell represents a geographic or logical region with its own local state.

2. **Infrastructure-ready**
   - First-class support for **energy grids, logistics networks, and public infrastructure**.
   - Deterministic, auditable behavior more important than pure TPS.

3. **Geography-aware consensus**
   - Validators are organized in **regional sets** and **cells**.
   - Cross-region coordination is explicit and modeled in the protocol.

4. **Separation of layers**
   - **Mesh & geography layer**
   - **Consensus & data availability**
   - **Execution & RWA tokenization**

---

## 2. Layered Model

### 2.1 Mesh Layer

- Defines the **global partition** of the network into:
  - **Regions**: high-level areas (e.g., South America, Europe, Asia…)
  - **Cells**: finer FEM-style subdivisions inside each region
- Each cell has:
  - `cell_id`
  - Coordinates / geospatial metadata
  - Assigned validator set (dynamic)
  - Local state root

The mesh is designed so that **adjacent cells** have well-defined interfaces, like FEM elements sharing boundaries.

### 2.2 Consensus & DA Layer

Responsible for:

- Block proposal and voting
- Finality rules
- Data availability guarantees

It operates on **global blocks** that contain:
- Cell-scoped transactions
- Cross-cell messages

### 2.3 Execution Layer

- Defines how **state transitions** are computed within each cell.
- Execution can be:
  - **Local** to a cell
  - **Coupled** to neighboring cells via boundary conditions (FEM analogy)
- Deterministic execution is crucial for:
  - Energy dispatch plans
  - Capacity allocation
  - Infrastructure reservations

### 2.4 RWA & Infrastructure Layer

- Top-level layer for:
  - Tokenized energy (MWh, capacity)
  - Infrastructure access rights
  - Logistic flows and contracts
- Uses **TrionChain Objects** (see `trion-objects.md`) to represent real-world positions and claims.

---

## 3. Topology Overview

At any point in time:

- The network maintains a **Mesh State**:
  - Set of regions
  - Set of cells per region
  - Mapping from cells → validator sets
- A **Global Block** references:
  - Previous global block hash
  - Merkle root of all cell states
  - Cross-cell message commitments

Graphically:

```text
[ Global Block n ]
   ├── Region A
   │     ├── Cell A1
   │     └── Cell A2
   ├── Region B
   │     ├── Cell B1
   │     └── Cell B2
   └── ...
