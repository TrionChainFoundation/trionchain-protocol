"""
sim_market_battery_distributed.py

Simulación simple (pero consistente) de:
- red de celdas (grid) con límites de línea (Pmax por arista)
- “market clearing” proxy (redispatch local + transferencias limitadas)
- congestión (utilización promedio de aristas)
- mismatch en un “corridor cut” (corte vertical)
- precios duales proxy (price = base + alpha*scarcity + beta*congestion)
- baterías distribuidas (en % de celdas), que cargan con excedente local y descargan con déficit local
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parámetros de modelo
# -----------------------------
@dataclass
class Params:
    nx: int = 8
    ny: int = 8
    steps: int = 60

    # Red (límites por línea, MW)
    line_pmax: float = 120.0

    # Demanda y generación base (MW por celda)
    base_load: float = 100.0
    base_gen: float = 100.0

    # Variabilidad temporal (perfil sinusoidal + ruido)
    load_amp: float = 35.0
    gen_amp: float = 35.0
    noise_sigma: float = 6.0

    # Precios (proxy)
    price_base: float = 10.0
    alpha_scarcity: float = 0.020   # impacto de unmet (MW) en precio
    beta_cong: float = 35.0         # impacto de congestión (0-1) en precio
    price_cap: float = 1000.0

    # Baterías distribuidas
    battery_share: float = 0.10     # 10% de celdas con batería
    Emax: float = 200.0             # MWh
    Pmax: float = 50.0              # MW
    eta_c: float = 0.95
    eta_d: float = 0.95
    soc0: float = 0.50              # fracción inicial de Emax

    # “Market clearing” proxy
    hops: int = 2                   # 2-hop neighborhood
    max_iter: int = 4               # iteraciones de balancing local (rápido)


# -----------------------------
# Utilidades de red
# -----------------------------
def idx(i, j, nx):
    return j * nx + i

def neighbors_4(i, j, nx, ny):
    # 4-conectividad
    for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < nx and 0 <= nj < ny:
            yield ni, nj

def build_edges(nx, ny):
    # Lista de aristas no dirigidas (u,v) entre vecinos 4
    edges = []
    for j in range(ny):
        for i in range(nx):
            u = idx(i,j,nx)
            for ni, nj in neighbors_4(i,j,nx,ny):
                v = idx(ni,nj,nx)
                if u < v:
                    edges.append((u, v))
    return edges

def corridor_cut_edges(nx, ny):
    """
    Corte vertical: entre columnas cut-1 y cut.
    Devuelve lista de aristas (u,v) que cruzan el corredor.
    """
    cut = nx // 2
    edges = []
    for j in range(ny):
        u = idx(cut-1, j, nx)
        v = idx(cut,   j, nx)
        edges.append((u, v))
    return edges

def hop_neighbors(node, nx, ny, hops=2):
    # BFS por hops para vecindario
    N = nx*ny
    visited = set([node])
    frontier = set([node])
    for _ in range(hops):
        new_frontier = set()
        for u in frontier:
            i = u % nx
            j = u // nx
            for ni, nj in neighbors_4(i, j, nx, ny):
                v = idx(ni, nj, nx)
                if v not in visited:
                    visited.add(v)
                    new_frontier.add(v)
        frontier = new_frontier
    visited.remove(node)
    return list(visited)


# -----------------------------
# Generación de perfiles (load/gen)
# -----------------------------
def make_profiles(p: Params, rng: np.random.Generator):
    N = p.nx * p.ny
    t = np.arange(p.steps)

    # componentes temporales globales
    load_t = p.base_load + p.load_amp * np.sin(2*np.pi*t / (p.steps/2.2))
    gen_t  = p.base_gen  + p.gen_amp  * np.sin(2*np.pi*(t+6) / (p.steps/2.2))

    # heterogeneidad espacial (celdas “más demandantes” y “más generadoras”)
    spatial_load = rng.uniform(0.85, 1.15, size=N)
    spatial_gen  = rng.uniform(0.85, 1.15, size=N)

    load = np.zeros((p.steps, N))
    gen  = np.zeros((p.steps, N))

    for k in range(p.steps):
        noise_L = rng.normal(0, p.noise_sigma, size=N)
        noise_G = rng.normal(0, p.noise_sigma, size=N)
        load[k] = np.clip(load_t[k] * spatial_load + noise_L, 0, None)
        gen[k]  = np.clip(gen_t[k]  * spatial_gen  + noise_G, 0, None)

    return load, gen


# -----------------------------
# Batería distribuida (local)
# -----------------------------
def battery_dispatch_local(net_inj, soc, has_batt, p: Params):
    """
    net_inj: (N,) MW  (gen - load) antes del mercado.
    soc: (N,) MWh
    has_batt: bool (N,)
    Retorna:
      net_inj2 (N,) MW
      soc2 (N,) MWh
      charge_MW (N,) MW (>=0)
      discharge_MW (N,) MW (>=0)
    """
    N = net_inj.size
    net2 = net_inj.copy()
    soc2 = soc.copy()
    charge = np.zeros(N)
    discharge = np.zeros(N)

    for n in range(N):
        if not has_batt[n]:
            continue

        # 1) Si hay excedente local -> cargar
        if net2[n] > 0:
            # potencia posible
            p_ch = min(p.Pmax, net2[n])  # MW
            # límite por energía disponible
            e_room = p.Emax - soc2[n]    # MWh
            p_ch = min(p_ch, e_room)     # asumiendo dt=1h
            if p_ch > 0:
                soc2[n] += p_ch * p.eta_c
                net2[n] -= p_ch
                charge[n] = p_ch

        # 2) Si hay déficit local -> descargar
        if net2[n] < 0:
            p_dis = min(p.Pmax, -net2[n])
            # energía disponible
            e_av = soc2[n]              # MWh
            p_dis = min(p_dis, e_av)    # dt=1h
            if p_dis > 0:
                soc2[n] -= p_dis
                net2[n] += p_dis * p.eta_d
                discharge[n] = p_dis

        # seguridad numérica
        soc2[n] = float(np.clip(soc2[n], 0.0, p.Emax))

    return net2, soc2, charge, discharge


# -----------------------------
# “Market clearing” proxy
# -----------------------------
def market_clearing_proxy(net_inj, nx, ny, line_pmax, hops=2, max_iter=4):
    """
    net_inj: (N,) MW, positivo = excedente, negativo = déficit.
    Simula balanceo por vecindarios 2-hop con límite de línea:
      - transfiere energía desde excedentes a déficits cercanos
      - genera “flows” agregados por aristas vecinas (simplificado)

    Retorna:
      traded_total (MW)  energía “movida”
      unmet_total (MW)   déficit no cubierto
      spill_total (MW)   excedente no usado
      edge_util_avg (0-1) utilización promedio proxy
      corridor_mismatch (MW) proxy del flujo neto a través del corte
    """
    N = net_inj.size
    surplus = np.clip(net_inj, 0, None)
    deficit = np.clip(-net_inj, 0, None)

    # Capacidad “efectiva” de transferencia por iteración (proxy de límites de líneas)
    # Cuanto más grande la red, más caminos; limitamos con line_pmax y conectividad.
    per_node_transfer_cap = line_pmax * 0.60

    # Construimos un proxy de “flows” por arista vecina
    edges = build_edges(nx, ny)
    edge_flow = {e: 0.0 for e in edges}

    traded = 0.0

    # iteraciones de matching local
    for _ in range(max_iter):
        # nodos con excedente
        sup_nodes = np.where(surplus > 1e-9)[0]
        if sup_nodes.size == 0:
            break

        for s in sup_nodes:
            if surplus[s] <= 1e-9:
                continue

            # límite de envío por nodo (proxy)
            send_cap = min(surplus[s], per_node_transfer_cap)
            if send_cap <= 0:
                continue

            neigh = hop_neighbors(s, nx, ny, hops=hops)
            if not neigh:
                continue

            # priorizar vecinos con déficit
            neigh_def = [n for n in neigh if deficit[n] > 1e-9]
            if not neigh_def:
                continue

            # repartir proporcional a déficit
            dvals = np.array([deficit[n] for n in neigh_def], dtype=float)
            dsum = float(dvals.sum())
            if dsum <= 0:
                continue

            # cuánta energía sale de s
            send = send_cap
            traded += send
            surplus[s] -= send

            # distribuir
            for n, dv in zip(neigh_def, dvals):
                take = send * (dv / dsum)
                take = min(take, deficit[n])
                deficit[n] -= take

                # registrar flow proxy: si n está a la derecha del corte, suma al corridor
                # y también asignamos flow a alguna arista “representativa” (vecina)
                # (simplificado: empuja flow hacia dirección dominante)
                si, sj = (s % nx), (s // nx)
                ni, nj = (n % nx), (n // nx)
                # elegimos una arista vecina aproximada (paso 1)
                step_i = si + np.sign(ni - si)
                step_j = sj + np.sign(nj - sj)
                step_i = int(np.clip(step_i, 0, nx-1))
                step_j = int(np.clip(step_j, 0, ny-1))
                u = idx(si, sj, nx)
                v = idx(step_i, step_j, nx)
                if u != v:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in edge_flow:
                        edge_flow[(a, b)] += take

    # métricas finales
    unmet = float(deficit.sum())
    spill = float(surplus.sum())

    # Utilización proxy promedio
    # normalizamos por line_pmax, y clip a [0,1]
    flows = np.array(list(edge_flow.values()), dtype=float)
    util = np.clip(np.abs(flows) / max(line_pmax, 1e-9), 0, 1)
    edge_util_avg = float(util.mean()) if util.size else 0.0

    # Mismatch en corredor (corte vertical)
    cut_edges = corridor_cut_edges(nx, ny)
    # proxy: suma de flujos asignados a aristas que cruzan corte, si existen en edge_flow
    corridor = 0.0
    for (u, v) in cut_edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_flow:
            corridor += edge_flow[(a, b)]
    corridor_mismatch = float(abs(corridor))

    return traded, unmet, spill, edge_util_avg, corridor_mismatch


# -----------------------------
# Precios (proxy dual)
# -----------------------------
def prices_from_metrics(unmet_cell, congestion, p: Params, rng: np.random.Generator):
    """
    unmet_cell: array (N,) MW de unmet por celda (proxy)
    congestion: escalar 0-1
    retorna prices (N,)
    """
    N = unmet_cell.size
    base = p.price_base + p.beta_cong * congestion
    # heterogeneidad de precios por celda
    eps = rng.normal(0, 0.35, size=N)
    price = base + p.alpha_scarcity * unmet_cell + eps
    return np.clip(price, 0, p.price_cap)


def run_simulation(p: Params, seed=7, use_battery=False):
    rng = np.random.default_rng(seed)
    N = p.nx * p.ny

    load, gen = make_profiles(p, rng)

    # baterías distribuidas
    has_batt = np.zeros(N, dtype=bool)
    soc = np.zeros(N, dtype=float)

    if use_battery:
        k = max(1, int(round(p.battery_share * N)))
        batt_nodes = rng.choice(N, size=k, replace=False)
        has_batt[batt_nodes] = True
        soc[has_batt] = p.soc0 * p.Emax

    # series
    traded_s = []
    unmet_s = []
    spill_s = []
    cong_s = []
    mismatch_s = []
    avg_price_s = []
    p10_s = []
    p90_s = []
    soc_avg_s = []
    ch_total_s = []
    dis_total_s = []

    for t in range(p.steps):
        net = gen[t] - load[t]  # MW

        # batería local antes del clearing
        if use_battery:
            net, soc, ch, dis = battery_dispatch_local(net, soc, has_batt, p)
            ch_total_s.append(float(ch.sum()))
            dis_total_s.append(float(dis.sum()))
        else:
            ch_total_s.append(0.0)
            dis_total_s.append(0.0)

        traded, unmet, spill, cong, mismatch = market_clearing_proxy(
            net, p.nx, p.ny, p.line_pmax, hops=p.hops, max_iter=p.max_iter
        )

        # Proxy “unmet por celda” para precios:
        # repartimos unmet global proporcional a carga (simple)
        total_load = float(load[t].sum()) + 1e-9
        unmet_cell = (unmet * (load[t] / total_load))

        prices = prices_from_metrics(unmet_cell, cong, p, rng)

        traded_s.append(traded)
        unmet_s.append(unmet)
        spill_s.append(spill)
        cong_s.append(cong)
        mismatch_s.append(mismatch)

        avg_price_s.append(float(np.mean(prices)))
        p10_s.append(float(np.percentile(prices, 10)))
        p90_s.append(float(np.percentile(prices, 90)))

        soc_avg_s.append(float(np.mean(soc)) if use_battery else 0.0)

    return {
        "traded": np.array(traded_s),
        "unmet": np.array(unmet_s),
        "spill": np.array(spill_s),
        "cong": np.array(cong_s),
        "mismatch": np.array(mismatch_s),
        "avg_price": np.array(avg_price_s),
        "p10": np.array(p10_s),
        "p90": np.array(p90_s),
        "soc_avg": np.array(soc_avg_s),
        "charge_total": np.array(ch_total_s),
        "discharge_total": np.array(dis_total_s),
    }


def main():
    p = Params()

    base = run_simulation(p, seed=7, use_battery=False)
    batt = run_simulation(p, seed=7, use_battery=True)

    x = np.arange(p.steps)

    # 1) Battery charge/discharge + SoC
    plt.figure()
    plt.plot(x, batt["charge_total"], label="Total Charge (MW)")
    plt.plot(x, batt["discharge_total"], label="Total Discharge (MW)")
    plt.title("Battery Charge/Discharge Power (distributed)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(x, batt["soc_avg"], label="Avg Battery SoC (MWh)")
    plt.title("Battery State of Charge (SoC)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MWh")
    plt.legend()
    plt.grid(True)

    # 2) Mismatch corridor cut
    plt.figure()
    plt.plot(x, base["mismatch"], label="Mismatch (baseline)")
    plt.plot(x, batt["mismatch"], label="Mismatch (with distributed battery)")
    plt.title("Boundary Mismatch (Flows) – LINE LIMITS + Market (corridor cut)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch (net flow across corridor) [MW]")
    plt.legend()
    plt.grid(True)

    # 3) Congestion
    plt.figure()
    plt.plot(x, base["cong"], label="Congestion (baseline)")
    plt.plot(x, batt["cong"], label="Congestion (with distributed battery)")
    plt.title("Network Congestion (avg edge utilization) - baseline vs distributed battery")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Utilization")
    plt.legend()
    plt.grid(True)

    # 4) Prices
    plt.figure()
    plt.plot(x, base["avg_price"], label="Avg Price (baseline)")
    plt.plot(x, batt["avg_price"], label="Avg Price (with battery)")
    plt.plot(x, base["p10"], label="P10 (baseline)")
    plt.plot(x, batt["p10"], label="P10 (with battery)")
    plt.plot(x, base["p90"], label="P90 (baseline)")
    plt.plot(x, batt["p90"], label="P90 (with battery)")
    plt.title("Prices (Dynamic + Congestion) - baseline vs distributed battery")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Price (dual units)")
    plt.legend()
    plt.grid(True)

    # 5) Unmet & Spill
    plt.figure()
    plt.plot(x, base["unmet"], label="Unmet (baseline)")
    plt.plot(x, batt["unmet"], label="Unmet (with battery)")
    plt.plot(x, base["spill"], label="Spill (baseline)")
    plt.plot(x, batt["spill"], label="Spill (with battery)")
    plt.title("Internal Market Clearing - Unmet & Spill (baseline vs distributed battery)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
