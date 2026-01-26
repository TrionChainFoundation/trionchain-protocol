# Current State and Roadmap

## Current State (What Is Already Built)

TrionChain is beyond the conceptual phase. The following components are already implemented and functional:

### Layer-1 Blockchain (Rust / Substrate)
- Custom Substrate-based node
- FEM-aware runtime structure
- TrionCell-based state model
- Authority-based block production
- Local development network running on VPS
- Blocks produced and finalized continuously

### Runtime & Pallets
- Dedicated pallets for physical state handling
- Structured storage for mesh-based state vectors
- Validation hooks designed for FEM constraint enforcement
- Deterministic execution model

### Oracle Layer (Off-Chain)
- Python-based oracle framework
- Structured physical state submission
- Mesh-aware oracle logic
- Simulated and real-data aviation use cases
- WebSocket-based communication with the node

### Research & Validation
- Extensive FEM simulations
- Physical-economic (PhyFi) models
- Stress, flow, and capacity validation experiments
- Documented technical whitepaper and brief

### Frontend & Visualization
- React-based dashboard
- Real-time chain connection
- Physical state visualization
- Oracle data inspection

---

## Current Limitations

While the core architecture exists, the following areas require further development:

- Formalized FEM constraint engine inside pallets
- Standardized oracle interface specification
- Expanded test coverage for physical invariants
- Multi-node testnet deployment
- External developer documentation and examples

---

## Grant-Funded Roadmap

### Phase 1 — FEM Constraint Engine (Core)
**Duration:** 2 months

- Implement full FEM constraint validation inside runtime
- Deterministic numerical handling
- Explicit rejection logic for infeasible states
- Runtime benchmarks and stress tests

---

### Phase 2 — Oracle Specification & SDK
**Duration:** 1.5 months

- Define canonical oracle message formats
- Authorization and permissioning rules
- Reference Python SDK
- Example real-world oracle integrations

---

### Phase 3 — Testnet & Validation
**Duration:** 1.5 months

- Multi-node TrionChain testnet
- Fault injection and adversarial testing
- Performance evaluation
- Public RPC endpoints

---

### Phase 4 — Documentation & Developer Access
**Duration:** 1 month

- Developer onboarding documentation
- Architecture diagrams
- End-to-end example (sensor → oracle → chain)
- Grant deliverables report

---

## Grant Impact

Grant funding will:
- Accelerate core protocol hardening
- Reduce oracle risk in RWA systems
- Enable verifiable physical settlement
- Produce a public, testable infrastructure-grade blockchain

The grant does **not** fund speculative research — it funds execution.

---

## Long-Term Vision

TrionChain aims to become the settlement layer for:
- Energy grids
- Transportation infrastructure
- Industrial assets
- Physical-financial hybrid systems (PhyFi)

This roadmap establishes the technical foundation for that future.
