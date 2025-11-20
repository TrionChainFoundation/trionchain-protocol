
markdown
# Consensus Model

This document describes the **consensus approach** for TrionChain at the conceptual level.

> Note: this is a **design draft**, not a finalized algorithm.

---

## 1. Objectives

- Provide **deterministic, auditable finality** suitable for infrastructure.
- Allow **geographically-aware validator sets**.
- Support **cell-scoped execution** with **global safety**.

---

## 2. Actors

- **Global Validators**
  - Participate in global block proposal and finality.
- **Regional Committees**
  - Subsets of validators focusing on a geographic region.
- **Cell Committees**
  - Subsets responsible for specific mesh cells.

A single validator can belong to multiple committees depending on stake and geography.

---

## 3. Two-Level Consensus (Conceptual)

1. **Cell Level (Local Agreement)**
   - Within each cell committee:
     - Propose **cell batches** of transactions.
     - Run a BFT-style agreement (e.g., HotStuff-like) to order and commit cell batches.
   - Output:
     - `cell_state_root`
     - `cell_batch_commitment`

2. **Global Level (Cross-Cell Agreement)**
   - Global validators collect:
     - Commitments from all active cells
     - Cross-cell messages
   - Construct a **Global Block**:
     - References the previous global block
     - Embeds a Merkle root of all `cell_state_root`s
     - Includes commitments for cross-cell messages
   - Run BFT-style finality over this global block.

---

## 4. Finality

- **Cell-level finality**: A cell batch is final once its local BFT round completes.
- **Global finality**: A global block is final once the global BFT round completes.
- Cross-cell effects become **globally final** only when:
  - The source cell batch is final, and
  - The global block including its cross-cell message commitment is final.

---

## 5. Fault Assumptions (Initial Draft)

- Each committee (cell, regional, global) assumes:
  - At most `f` Byzantine validators out of `3f + 1` total in that committee.
- Committees can overlap; safety analysis must consider:
  - Correlation of faults across regions
  - Network partitions along geographic lines

---

## 6. Open Questions

- Formal mapping between **stake distributions** and **cell committee composition**.
- Latency / throughput trade-offs between:
  - Cell-only rounds
  - Global aggregation rounds
- Integration with **data availability sampling** for large meshes.
