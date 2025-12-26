#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-country energy mesh simulation (A exports to B)
- Explicit cross-border flow A → B across discrete border corridors
- Settlement paid by B in external token (PAY)
- Protocol fee paid in native token (TRN), where fee follows *physical stress* (corridor utilization)

Outputs (PNG):
- country_A_price_mean/std/last.png
- country_B_price_mean/std/last.png
- two_country_price_mean_frontier_stress.png  (red/blue heatmap + hot frontier)
- two_country_stress_std_frontier_volatility.png
- cross_border_trade_flow_A_to_B.png
- settlement_cost_paid_by_B_PAY.png
- protocol_fees_paid_TRN.png
- border_corridor_utilization_heatmap.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
SEED = 7
np.random.seed(SEED)

NA = 20          # Country A grid width  (x)
NB = 20          # Country B grid width  (x)
NY = 20          # shared height (y)
STEPS = 60

# Border: A right edge connects to B left edge via multiple "corridors" along y
# We model a subset of y-rows as corridors (others have near-zero capacity)
CORRIDOR_ROWS = [0, 3, 8, 11, 14, 19]  # pick any rows you want
BASE_BORDER_CAP = 0.12                # MWh per step per corridor (baseline)
NON_CORRIDOR_CAP = 0.01               # tiny leakage elsewhere

# Intra-country neighbor transfer capacity (simplified diffusion / transport)
INTRA_CAP_A = 0.25
INTRA_CAP_B = 0.20

# Price model parameters
BASE_PRICE_A = 0.90
BASE_PRICE_B = 2.50
ALPHA_SCARCITY = 1.60   # price sensitivity to unmet demand
BETA_CONGEST  = 0.90    # price sensitivity to congestion/stress

# Settlement + fees
PAY_PER_MWH = 2.20      # B pays A in PAY tokens per MWh imported (external token)
TRN_FEE_RATE = 0.06     # protocol fee rate in TRN per unit stress-weighted import (see below)
TRN_FEE_EXP = 2.0       # nonlinear amplification: higher stress => much higher fee

# Output folder
OUTDIR = "results_two_country"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# Helper: 2D neighbor exchange (very simplified “mesh coupling”)
# ---------------------------
def neighbor_exchange(net, cap):
    """
    net[y,x] > 0 means surplus energy to push out
    net[y,x] < 0 means deficit energy wanting inflow
    We do a single pass pushing from surplus cells to deficit neighbors, limited by cap.
    """
    ny, nx = net.shape
    flow = np.zeros_like(net)

    # 4-neighborhood: up/down/left/right
    neigh = [(-1,0),(1,0),(0,-1),(0,1)]

    for y in range(ny):
        for x in range(nx):
            if net[y, x] <= 0:
                continue
            surplus = net[y, x]
            # collect deficit neighbors
            deficits = []
            for dy, dx in neigh:
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and net[yy, xx] < 0:
                    deficits.append((yy, xx, -net[yy, xx]))
            if not deficits:
                continue

            total_need = sum(d for _,_,d in deficits)
            # total outflow limited
            out = min(surplus, cap, total_need)

            for yy, xx, need in deficits:
                share = out * (need / total_need)
                flow[y, x] -= share
                flow[yy, xx] += share

    return flow  # net update to apply


# ---------------------------
# Initialize fields
# ---------------------------
# Generation and demand patterns (toy)
# A has more generation capacity, B has more demand.
genA0 = 1.40 + 0.20*np.random.rand(NY, NA)
demA0 = 1.05 + 0.25*np.random.rand(NY, NA)

genB0 = 0.85 + 0.15*np.random.rand(NY, NB)
demB0 = 1.20 + 0.30*np.random.rand(NY, NB)

# Add some spatial structure: A gen higher on left, B demand higher in center
xA = np.linspace(0, 1, NA)[None, :]
xB = np.linspace(0, 1, NB)[None, :]
y  = np.linspace(0, 1, NY)[:, None]

genA0 += 0.20*(1 - xA)
demB0 += 0.20*np.exp(-((xB-0.5)**2 + (y-0.5)**2)/0.12)

# Battery (optional): distributed storage that reduces volatility (very simple)
socA = 0.35*np.ones((NY, NA))
socB = 0.25*np.ones((NY, NB))
BAT_CAP = 1.0
BAT_PMAX = 0.35

# Storage behavior parameters
CHARGE_WHEN_SURPLUS = True
DISCHARGE_WHEN_DEFICIT = True

# Border capacities per row
border_cap = np.full(NY, NON_CORRIDOR_CAP)
border_cap[CORRIDOR_ROWS] = BASE_BORDER_CAP

# ---------------------------
# Time series storage
# ---------------------------
pricesA_hist = []
pricesB_hist = []
border_util_hist = []   # [step, corridor_index] utilization ratio
border_flow_hist = []   # [step, corridor_index] MWh
total_import_hist = []  # total MWh A→B per step

settlement_pay_hist = []  # PAY tokens per step
fee_trn_hist = []         # TRN per step

# ---------------------------
# Main simulation loop
# ---------------------------
for t in range(STEPS):
    # Introduce mild time-varying shocks
    shock = 0.10*np.sin(2*np.pi*t/30.0)
    genA = genA0 * (1.0 + 0.05*np.cos(2*np.pi*t/40.0))
    demA = demA0 * (1.0 + 0.03*np.sin(2*np.pi*t/25.0))

    genB = genB0 * (1.0 + 0.03*np.cos(2*np.pi*t/35.0))
    demB = demB0 * (1.0 + 0.06*np.sin(2*np.pi*t/28.0) + shock)

    # Net power balance (positive surplus, negative deficit)
    netA = genA - demA
    netB = genB - demB

    # ---------------------------
    # Battery actions (toy): charge on surplus, discharge on deficit
    # ---------------------------
    if CHARGE_WHEN_SURPLUS:
        chargeA = np.clip(netA, 0, BAT_PMAX)
        roomA = BAT_CAP - socA
        chargeA = np.minimum(chargeA, roomA)
        socA += chargeA
        netA -= chargeA

        chargeB = np.clip(netB, 0, BAT_PMAX)
        roomB = BAT_CAP - socB
        chargeB = np.minimum(chargeB, roomB)
        socB += chargeB
        netB -= chargeB

    if DISCHARGE_WHEN_DEFICIT:
        dischargeA = np.clip(-netA, 0, BAT_PMAX)
        dischargeA = np.minimum(dischargeA, socA)
        socA -= dischargeA
        netA += dischargeA

        dischargeB = np.clip(-netB, 0, BAT_PMAX)
        dischargeB = np.minimum(dischargeB, socB)
        socB -= dischargeB
        netB += dischargeB

    # ---------------------------
    # Intra-country neighbor exchange (simple local redistribution)
    # ---------------------------
    netA += neighbor_exchange(netA, INTRA_CAP_A)
    netB += neighbor_exchange(netB, INTRA_CAP_B)

    # ---------------------------
    # Cross-border exchange A → B (explicit)
    # A exports only if A border cell has surplus AND B border cell has deficit.
    # ---------------------------
    # Border cells: A at x=NA-1, B at x=0
    flows_per_corridor = []
    utils_per_corridor = []
    total_import = 0.0

    for r in CORRIDOR_ROWS:
        cap = border_cap[r]
        a_surplus = max(netA[r, NA-1], 0.0)
        b_need    = max(-netB[r, 0],  0.0)

        f = min(a_surplus, b_need, cap)  # MWh imported by B from A on this corridor
        # apply exchange
        netA[r, NA-1] -= f
        netB[r, 0]    += f

        util = (f / cap) if cap > 0 else 0.0
        flows_per_corridor.append(f)
        utils_per_corridor.append(util)
        total_import += f

    # Save corridor traces
    border_flow_hist.append(flows_per_corridor)
    border_util_hist.append(utils_per_corridor)
    total_import_hist.append(total_import)

    # ---------------------------
    # Compute scarcity + congestion metrics
    # ---------------------------
    unmetA = np.clip(-netA, 0, None)
    unmetB = np.clip(-netB, 0, None)

    # “Stress” proxy: unmet demand + local gradient (congestion)
    # (toy) use absolute net after exchanges as a proxy for local imbalance
    stressA = unmetA + 0.25*np.abs(netA)
    stressB = unmetB + 0.25*np.abs(netB)

    # border stress: high when utilization is high
    # map it onto the border columns
    border_stress_colA = np.zeros(NY)
    border_stress_colB = np.zeros(NY)
    for idx, r in enumerate(CORRIDOR_ROWS):
        u = utils_per_corridor[idx]
        border_stress_colA[r] = u
        border_stress_colB[r] = u

    # ---------------------------
    # Price fields (toy):
    # price = base + scarcity term + congestion term
    # ---------------------------
    priceA = BASE_PRICE_A + ALPHA_SCARCITY*unmetA + BETA_CONGEST*stressA
    priceB = BASE_PRICE_B + ALPHA_SCARCITY*unmetB + BETA_CONGEST*stressB

    # Add explicit border premium (stress concentrates at frontier)
    priceA[:, NA-1] += 0.35*border_stress_colA
    priceB[:, 0]    += 0.45*border_stress_colB

    pricesA_hist.append(priceA.copy())
    pricesB_hist.append(priceB.copy())

    # ---------------------------
    # Settlement + TRN fee (fee follows stress)
    # ---------------------------
    settlement_pay = PAY_PER_MWH * total_import  # B pays A in PAY
    # Stress-weighted fee: higher corridor utilization => superlinear fee
    util_arr = np.array(utils_per_corridor)  # in [0,1]
    stress_factor = np.mean(util_arr**TRN_FEE_EXP) if util_arr.size else 0.0
    fee_trn = TRN_FEE_RATE * total_import * (1.0 + 6.0*stress_factor)

    settlement_pay_hist.append(settlement_pay)
    fee_trn_hist.append(fee_trn)

# Convert histories to arrays
pricesA_hist = np.array(pricesA_hist)   # [T, NY, NA]
pricesB_hist = np.array(pricesB_hist)   # [T, NY, NB]
border_util_hist = np.array(border_util_hist)  # [T, nCorr]
border_flow_hist = np.array(border_flow_hist)  # [T, nCorr]
total_import_hist = np.array(total_import_hist)

settlement_pay_hist = np.array(settlement_pay_hist)
fee_trn_hist = np.array(fee_trn_hist)

# ---------------------------
# Aggregations
# ---------------------------
A_mean = pricesA_hist.mean(axis=0)
A_std  = pricesA_hist.std(axis=0)
A_last = pricesA_hist[-1]

B_mean = pricesB_hist.mean(axis=0)
B_std  = pricesB_hist.std(axis=0)
B_last = pricesB_hist[-1]

# Two-country stitched maps for presentation (A | frontier | B)
# Frontier column will be a synthetic “stress” column based on average utilization by row
frontier_col_mean = np.zeros((NY, 1))
frontier_col_std  = np.zeros((NY, 1))

# Map corridor utilization to frontier column
util_by_row_mean = np.zeros(NY)
util_by_row_std  = np.zeros(NY)
for i, r in enumerate(CORRIDOR_ROWS):
    util_by_row_mean[r] = border_util_hist[:, i].mean()
    util_by_row_std[r]  = border_util_hist[:, i].std()

frontier_col_mean[:, 0] = util_by_row_mean
frontier_col_std[:, 0]  = util_by_row_std

two_mean = np.concatenate([A_mean, frontier_col_mean, B_mean], axis=1)
two_std  = np.concatenate([A_std,  frontier_col_std,  B_std ], axis=1)

# ---------------------------
# Plot helpers
# ---------------------------
def save_heatmap(mat, title, cbar_label, path, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label=cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def save_two_country_heatmap(mat, title, cbar_label, path, frontier_x, cmap="coolwarm"):
    plt.figure(figsize=(12, 4))
    plt.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    plt.title(title)
    plt.xlabel("Country A  |  Frontier  |  Country B")
    plt.ylabel("TrionCell row")
    plt.colorbar(label=cbar_label)
    # vertical line marking frontier column
    plt.axvline(frontier_x, linewidth=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# ---------------------------
# Save figures: per-country
# ---------------------------
save_heatmap(A_mean, "Country A (Exporter) — Price map (mean over time)", "Price / Stress (arb.)",
             os.path.join(OUTDIR, "country_A_price_mean.png"))
save_heatmap(A_std,  "Country A (Exporter) — Price map (std over time)",  "Price / Stress (arb.)",
             os.path.join(OUTDIR, "country_A_price_std.png"))
save_heatmap(A_last, "Country A (Exporter) — Price map (last step)",     "Price / Stress (arb.)",
             os.path.join(OUTDIR, "country_A_price_last.png"))

save_heatmap(B_mean, "Country B (Importer) — Price map (mean over time)", "Price / Stress (arb.)",
             os.path.join(OUTDIR, "country_B_price_mean.png"))
save_heatmap(B_std,  "Country B (Importer) — Price map (std over time)",  "Price / Stress (arb.)",
             os.path.join(OUTDIR, "country_B_price_std.png"))
save_heatmap(B_last, "Country B (Importer) — Price map (last step)",      "Price / Stress (arb.)",
             os.path.join(OUTDIR, "country_B_price_last.png"))

# ---------------------------
# Save figures: presentation stitched heatmaps (red/blue)
# ---------------------------
frontier_x = NA  # because A has columns 0..NA-1, frontier column index is NA
save_two_country_heatmap(
    two_mean,
    "Two-Country Price Map (Mean Over Time) — Frontier Stress",
    "Price / Stress (arb. units)",
    os.path.join(OUTDIR, "two_country_price_mean_frontier_stress.png"),
    frontier_x=frontier_x,
    cmap="coolwarm"
)

save_two_country_heatmap(
    two_std,
    "Two-Country Stress Map (Std Over Time) — Frontier Volatility",
    "Price / Stress (arb. units)",
    os.path.join(OUTDIR, "two_country_stress_std_frontier_volatility.png"),
    frontier_x=frontier_x,
    cmap="coolwarm"
)

# ---------------------------
# Cross-border trade flow over time
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(total_import_hist)
plt.title("Cross-border trade flow A→B (MWh per step)")
plt.xlabel("Step")
plt.ylabel("MWh")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cross_border_trade_flow_A_to_B.png"), dpi=180)
plt.close()

# ---------------------------
# Settlement cost in PAY over time
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(settlement_pay_hist)
plt.title("Settlement cost paid by B (PAY token, per step)")
plt.xlabel("Step")
plt.ylabel("PAY")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "settlement_cost_paid_by_B_PAY.png"), dpi=180)
plt.close()

# ---------------------------
# Protocol fees in TRN over time (fee follows stress)
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(fee_trn_hist)
plt.title("Protocol fees paid in TRN (per step)")
plt.xlabel("Step")
plt.ylabel("TRN")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "protocol_fees_paid_TRN.png"), dpi=180)
plt.close()

# ---------------------------
# Corridor utilization heatmap (each corridor over time)
# ---------------------------
plt.figure(figsize=(10, 4))
plt.imshow(border_util_hist.T, origin="lower", aspect="auto", cmap="viridis")
plt.title("Border corridor utilization (each corridor over time)")
plt.xlabel("Step")
plt.ylabel("Corridor index")
plt.colorbar(label="Utilization (0..1)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "border_corridor_utilization_heatmap.png"), dpi=180)
plt.close()

print(f"Done. Figures saved in: {OUTDIR}")
print(f"Corridor rows: {CORRIDOR_ROWS}")
print("Tip: upload the PNGs to GitHub and embed the stitched red/blue frontier map in your deck cover.")
