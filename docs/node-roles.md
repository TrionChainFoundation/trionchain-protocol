# Node Roles

TrionChain defines several **node classes**, each with a specific role in the protocol.

---

## 1. Core Roles

### 1.1 Global Validator

- Participates in **global block** consensus.
- Holds stake and voting power at the protocol level.
- May also be part of multiple regional / cell committees.

### 1.2 Regional Validator

- Focused on a specific **region** (set of cells).
- Helps:
  - Propose and validate cell batches
  - Maintain high availability in that geography

### 1.3 Cell Validator

- Specialized in one or more **cells**.
- Responsible for:
  - Proposing cell batches
  - Executing cell transactions
  - Producing `cell_state_root` commitments

---

## 2. Supporting Roles

### 2.1 Observer / Light Node

- Stores headers and proofs instead of full state.
- Verifies:
  - Global blocks
  - Cell commitments
- Suitable for:
  - Wallets
  - Regulatory observers
  - Infrastructure operators who just need verification.

### 2.2 Bridge Node

- Handles interoperability with external chains.
- Converts:
  - TrionChain objects ↔ external token formats
- Must follow stricter security and auditing.

### 2.3 RWA Oracle

- Feeds off-chain infrastructure data:
  - Measurements, sensor data, grid status, etc.
- Publishes data as **signed reports** that:
  - Are referenced by on-chain contracts
  - May be tied to specific cells / regions.

---

## 3. Future Work

- Detailed **slashing rules** and incentives per role.
- Formal definition of **eligibility** and **rotation** for cell committees.
