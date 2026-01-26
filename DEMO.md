# TrionChain DevNet Demo (Polkadot.js)

This document provides a **reproducible demo** for reviewers to verify that TrionChain is:
- producing blocks on a live DevNet
- exposing RPC publicly
- enforcing runtime-level validation logic (accept/reject)

---

## 1) DevNet Endpoint

**WebSocket RPC (public):**
- `ws://45.55.93.146:9944`

If you are using an SSH tunnel from your laptop:
- `ws://127.0.0.1:9944`

> Note: The node is running as a systemd service on the server.

---

## 2) Open Polkadot.js Apps

1. Go to Polkadot.js Apps
2. Click the network selector (top-left)
3. Choose **Development** → **Custom**
4. Paste the endpoint:
   - `ws://45.55.93.146:9944`
5. Click **Switch**

You should immediately see:
- new blocks being produced
- a live chain connection status

---

## 3) Confirm Block Production (Server-side)

On the server:

```bash
sudo systemctl status trionchain-devnet
journalctl -u trionchain-devnet -f
