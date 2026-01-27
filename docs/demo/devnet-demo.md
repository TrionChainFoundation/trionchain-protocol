# TrionChain DevNet – Technical Demo

## Overview

This document describes the current TrionChain DevNet demo running on a
Substrate-based solo chain with a custom runtime and FEM-inspired pallet.

The goal of this demo is to prove:
- Custom runtime deployment
- Custom pallet integration
- On-chain extrinsics execution
- Verified on-chain events

---

## Network Details

- Chain name: TrionChain DevNet
- Node: Substrate Solochain
- Block time: ~6 seconds
- RPC: 9944
- Runtime version: 0.1.0
- Consensus: Aura + Grandpa
- Environment: Ubuntu 24.04 (VM)

---

## Custom Pallet: `trionFem`

The `trionFem` pallet represents the first FEM-inspired building block of TrionChain.

It introduces the concept of **cells** (finite regions) that can be:
- Identified by `cellId`
- Assigned an authorized oracle/owner
- Managed via governance (Root)

This pallet is included directly in the runtime and exposed via Polkadot.js.

---

## Executed Extrinsic (Demo)

A sudo call was executed to assign an oracle/owner to a FEM cell.

### Call
sudo → trionFem.setCellOwner(cellId: 0, owner: ALICE)


### Result
- Extrinsic successfully executed
- On-chain event emitted

---

## On-chain Event

The following event was observed on-chain:

trionFem.CellOwnerSet


**Parameters:**
- cellId: 0
- owner: ALICE

This confirms:
- Pallet logic execution
- Storage mutation
- Event emission
- Runtime correctness

---

## Verification

### Metadata
The pallet is present in runtime metadata and detectable via RPC:
- `state_getMetadata`
- Pallet name: `trionFem`

### Polkadot.js UI
- Pallet visible under Developer → Extrinsics
- Pallet callable via Sudo
- Events visible in Explorer

---

## Current Status

- DevNet producing blocks ✔
- Custom runtime deployed ✔
- Custom pallet integrated ✔
- Extrinsics working ✔
- Events emitted ✔

This demo establishes the technical foundation for further FEM-based
computation, oracle integration, and physical system modeling.

---

## Next Steps

- Add non-sudo extrinsics
- Extend cell state (values, meshes, constraints)
- Introduce permissionless updates
- Progress toward multi-cell FEM computation

