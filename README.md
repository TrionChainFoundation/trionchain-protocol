# TrionChain – Protocol Specification

This repository contains the **core protocol specification** for the TrionChain FEM-based blockchain – a geographically-aware, infrastructure-oriented L1 whose execution and data layout are inspired by the **Finite Element Method (FEM)**.

The goal of this repo is to serve as the **single source of truth** for:
- Architecture & network topology
- Consensus and validator roles
- FEM-style computation model
- Data model and transaction lifecycle
- RWA & infrastructure integration layer

---

## Repository Map

- [`docs/architecture.md`](docs/architecture.md)  
  High-level architecture, layers and mesh topology.

- [`docs/consensus.md`](docs/consensus.md)  
  Consensus model, finality, and regional / cell coordination.

- [`docs/fem-computation.md`](docs/fem-computation.md)  
  How TrionChain maps FEM concepts (mesh, elements, degrees of freedom) to blockchain state and computation.

- [`docs/node-roles.md`](docs/node-roles.md)  
  Validator classes, cell coordinators, observers, bridges, and oracles.

- [`docs/rwa-layer.md`](docs/rwa-layer.md)  
  Tokenization layer for energy, logistics, and real-world infrastructure.

- [`docs/trion-objects.md`](docs/trion-objects.md)  
  Core on-chain objects and token types (TON, infrastructure tokens, cell assets).

- [`specs/protocol-overview.md`](specs/protocol-overview.md)  
  State machine, blocks, transactions, and high-level safety assumptions.

- [`specs/data-model.md`](specs/data-model.md)  
  Block header, cell metadata, and storage layout.

- [`specs/transaction-lifecycle.md`](specs/transaction-lifecycle.md)  
  From client request → routing to a cell → inclusion → finality.

- [`research/reading-list.md`](research/reading-list.md)  
  Background material (FEM, geospatial systems, BFT consensus, RWA).

For the **public conceptual description**, see the separate repository:

> 🔗 `TrionChainFoundation/public-whitepaper`  
> “TrionChain – Public Whitepaper (2025)”

---

## High-Level Vision

TrionChain aims to:

- Use **FEM-like meshes** to partition the world into infrastructure-aware “cells”
- Offer **deterministic, geographically-aware execution** for energy, logistics, and national-scale systems
- Provide a clean separation between:
  - **Protocol layer** (this repository)
  - **Public narrative / whitepaper** (`public-whitepaper`)
  - **Implementation repos** (nodes, SDKs, tooling – to be created)

---

## Roadmap (Specification Side)

1. **🔹 v0.1 – Conceptual Spec (current)**
   - Architecture, consensus, FEM model, and data model drafts
2. **🔹 v0.2 – Formalization**
   - Pseudocode, message flows, and safety / liveness assumptions
3. **🔹 v0.3 – Implementation-Ready Spec**
   - Node roles, networking, storage APIs, and testing guidelines

---

## Contributing

At this stage, the spec is **exploratory and evolving**.

- Issues: protocol questions, modeling ideas, research links  
- Pull Requests: corrections, clarifications, improved diagrams or examples

In the future, TrionChainFoundation will define a formal **governance and RFC process** for major protocol changes.

---

## License

Unless otherwise noted, the contents of this repository are released under the **MIT License**.

