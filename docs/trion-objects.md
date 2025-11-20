# TrionChain Objects

TrionChain uses a set of **core on-chain objects** to represent state, RWA, and infrastructure flows.

This is an **early conceptual list**.

---

## 1. TON – Trion Object Node (Working Name)

- Base unit representing a **position in the TrionChain mesh**.
- Carries:
  - `cell_id`
  - Optional coordinates / metadata
  - Links to other objects

TONs are used as the **anchors** for more complex objects.

---

## 2. InfrastructureAsset

Represents a physical asset, for example:

- Power line segment
- Transformer
- Pipeline segment
- Road / rail segment

Fields (conceptual):

- `asset_id`
- `cell_id` (or list of cells for multi-cell assets)
- `type`
- `capacity`
- `owner`
- `regulatory_flags`

---

## 3. CapacitySlot

Represents a **time-bounded reservation** on an asset.

- `slot_id`
- `asset_id`
- `time_window`
- `max_capacity`
- `holder`

These can be:
- Traded
- Reserved
- Used as collateral for infrastructure contracts.

---

## 4. Flow / Shipment Objects

For energy or goods moving through the mesh:

- `flow_id`
- Path (list of assets / cells)
- Quantity
- Time profile
- Owner / counter-parties

---

## 5. Token Layer

On top of objects, standard token interfaces can be built:

- Fungible tokens representing:
  - kWh, MWh
  - Transport capacity units
- Non-fungible tokens representing:
  - Specific infrastructure rights
  - Long-term concessions.

Implementation details will live in future **smart contract repos**; here we focus on the **protocol-level representation**.
