"""
crear_matrices_transbordo.py

Módulo para generar matrices de TRANSBORDO del pipeline cubo izquierdo.
Combina transbordos explícitos (x_pred_checked[tipo == 'TRANSBORDO'])
e implícitos (self.transbordos_implicitos) para producir dos vistas:

  - Macro: cuántos transbordos hay por estación (top N) en intervalos de 15 min.
  - Micro: detalle línea de llegada / línea de salida / sentido por intervalo de 15 min.

Cobertura de red:
  - Metro (02_, A9_) y cercanías (04_): lookup en rutas_transbordo.csv
  - Bus EMT (03_): lookup directo en topologia_solo_EMT.csv por DPAYPOINT
  - Bus interurbano (resto): lookup directo en topologia_sin_EMT.csv por DPAYPOINT
  En buses cada DPAYPOINT identifica unívocamente línea y sentido → no hay fallback.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
TOPO_DIR = BASE_DIR.parent / "topologia"
CHECK_DIR = BASE_DIR / "check_transbordo"
EMT_TOPO_PATH  = BASE_DIR / "prediccion_descenso" / "topologia_solo_EMT.csv"
INTER_TOPO_PATH = BASE_DIR / "prediccion_descenso" / "topologia_sin_EMT.csv"

_METRO_CERC_PREFIXES = ("02_", "A9_", "04_")

_EMT_PREFIX = "03_"

CIRCULAR_SENTIDOS = {
    "L6":  ("LUCERO", "LAGUNA"),
    "L12": ("ALCORCON CENTRAL", "PARQUE DE LISBOA"),
}

def _build_ap_index(G):
    """
    Construye un índice {punto_acceso: set(nodos)} a partir del atributo
    'access_point_codes' de cada nodo del MultiDiGraph.
    """
    idx = defaultdict(set)
    for node, data in G.nodes(data=True):
        aps = data.get("access_point_codes", set())
        if isinstance(aps, str):
            aps = {aps}
        for ap in aps:
            idx[ap].add(node)
    return dict(idx)

def _get_line_sequence(line_adj: dict, start: str) -> list:
    """
    Recorre la línea hacia adelante desde 'start' siguiendo las aristas dirigidas
    del grafo filtrado a esa línea, y devuelve la secuencia ordenada de nodos.
    Se detiene cuando no hay más nodos no visitados.
    """
    seq = [start]
    visited = {start}
    current = start
    while True:
        nexts = [n for n in line_adj.get(current, []) if n not in visited]
        if not nexts:
            break
        current = nexts[0]
        seq.append(current)
        visited.add(current)
    return seq

def _process_circular_line(adj, node_aps, line, sentido_a, sentido_b) -> list:
    """
    Genera todas las tuplas (origen_ap, destino_ap, linea, sentido, num_paradas)
    para una línea circular usando dos estaciones de referencia como etiquetas
    de sentido.

    Estrategia:
      - Se recorre el círculo desde sentido_a con DFS, obteniendo la secuencia
        de estaciones en una dirección. Todos los pares (i, j) con j > i en esa
        secuencia tienen sentido = sentido_b (van "hacia sentido_b").
      - La secuencia inversa representa la dirección opuesta. Todos sus pares
        tienen sentido = sentido_a (van "hacia sentido_a").
    De este modo se cubren los N*(N-1) pares dirigidos del círculo.

    Parámetros
    ----------
    adj : dict
        Adyacencia dirigida de la línea: {nodo: [vecinos_salientes]}.
    node_aps : dict
        Índice nodo → set(access_point_codes).
    line : str
        Identificador de la línea (ej. 'L6', 'L12').
    sentido_a, sentido_b : str
        Nombres de los dos nodos de referencia que etiquetan los sentidos.

    Retorna
    -------
    list de dicts con claves:
        punto_acceso_origen, punto_acceso_destino, linea_a_usar, sentido,
        num_paradas_hasta_destino.
    """
    all_circle_nodes = set(adj.keys()) | {v for nbrs in adj.values() for v in nbrs}
    if sentido_a not in all_circle_nodes:
        return []

    seq_forward = _get_line_sequence(dict(adj), sentido_a)

    seq_backward = seq_forward[::-1]

    rows = []
    for seq, sentido in [(seq_forward, sentido_b), (seq_backward, sentido_a)]:
        for i, src_node in enumerate(seq[:-1]):
            src_aps = list(node_aps.get(src_node, set()))
            if not src_aps:
                continue
            for j in range(i + 1, len(seq)):
                dst_node = seq[j]
                dst_aps = list(node_aps.get(dst_node, set()))
                if not dst_aps:
                    continue
                num_paradas = j - i
                for src_ap in src_aps:
                    for dst_ap in dst_aps:
                        rows.append({
                            "punto_acceso_origen": src_ap,
                            "punto_acceso_destino": dst_ap,
                            "linea_a_usar": line,
                            "sentido": sentido,
                            "num_paradas_hasta_destino": num_paradas,
                        })
    return rows

def _process_graph(G) -> list:
    """
    Extrae todas las tuplas (origen_ap, destino_ap, linea, sentido, num_paradas)
    de un MultiDiGraph de metro o cercanías.

    El grafo es bidireccional: cada par de estaciones consecutivas tiene arista
    en ambos sentidos. Los terminales reales son los nodos con exactamente
    1 vecino único. Se arranca desde cada terminal y el BFS recorre la línea
    hacia el otro extremo (el nodo previo queda visitado, forzando la dirección).

    Para cada línea:
      1. Detecta los terminales (nodos con 1 vecino único).
      2. Arranca desde cada terminal y genera la secuencia ordenada.
      3. El nodo final de la secuencia es el 'sentido' del viaje.
      4. Genera todos los pares (i, j) con j > i en la secuencia.

    Devuelve lista de dicts con claves:
      punto_acceso_origen, punto_acceso_destino, linea_a_usar, sentido,
      num_paradas_hasta_destino.
    """

    line_adj = defaultdict(lambda: defaultdict(list))
    for u, v, data in G.edges(data=True):
        line = data.get("line_id")
        if line:
            line_adj[line][u].append(v)

    node_aps = {}
    for node, data in G.nodes(data=True):
        aps = data.get("access_point_codes", set())
        if isinstance(aps, str):
            aps = {aps}
        node_aps[node] = aps

    rows = []

    for line, adj in line_adj.items():
        all_nodes = set(adj.keys()) | {v for nbrs in adj.values() for v in nbrs}

        unique_neighbors = {n: set() for n in all_nodes}
        for u, nbrs in adj.items():
            for v in nbrs:
                unique_neighbors[u].add(v)
                unique_neighbors[v].add(u)

        terminals = sorted(n for n, nbrs in unique_neighbors.items() if len(nbrs) == 1)

        if not terminals:

            if line in CIRCULAR_SENTIDOS:
                sentido_a, sentido_b = CIRCULAR_SENTIDOS[line]
                rows.extend(_process_circular_line(dict(adj), node_aps, line, sentido_a, sentido_b))

            continue

        seen = set()
        for start in terminals:
            seq = _get_line_sequence(dict(adj), start)
            if len(seq) < 2:
                continue

            seq_key = (seq[0], seq[-1])
            if seq_key in seen:
                continue
            seen.add(seq_key)

            terminal_name = seq[-1]

            for i, src_node in enumerate(seq[:-1]):
                src_aps = list(node_aps.get(src_node, set()))
                if not src_aps:
                    continue

                for j in range(i + 1, len(seq)):
                    dst_node = seq[j]
                    dst_aps = list(node_aps.get(dst_node, set()))
                    if not dst_aps:
                        continue

                    num_paradas = j - i
                    for src_ap in src_aps:
                        for dst_ap in dst_aps:
                            rows.append({
                                "punto_acceso_origen": src_ap,
                                "punto_acceso_destino": dst_ap,
                                "linea_a_usar": line,
                                "sentido": terminal_name,
                                "num_paradas_hasta_destino": num_paradas,
                            })

    return rows

def crear_csv_rutas_transbordo(
    out_path=None,
    metro_pkl_path=None,
    cercanias_pkl_path=None,
    tabla_paradas_path=None,
):
    """
    Genera un CSV de referencia que, para cada par (origen, destino) de metro
    y cercanías, indica qué línea usar, el sentido del viaje (estación terminal
    en esa dirección) y el número de paradas.

    Este CSV es reutilizable entre ejecuciones del pipeline. Solo es necesario
    regenerarlo si cambia la topología de la red (los .pkl).

    Parámetros
    ----------
    out_path : Path or str, optional
        Ruta donde guardar el CSV. Si es None, solo se devuelve el DataFrame.
    metro_pkl_path : Path or str, optional
        Ruta al pickle del grafo de metro. Por defecto:
        check_transbordo/metro_multidigraph.pkl
    cercanias_pkl_path : Path or str, optional
        Ruta al pickle del grafo de cercanías. Por defecto:
        check_transbordo/cercanias_multidigraph_sin_civis.pkl
    tabla_paradas_path : Path or str, optional
        Ruta a tabla_puntos_acceso_con_coords.csv para enriquecer con nombres.
        Por defecto: topologia/tabla_puntos_acceso_con_coords.csv

    Retorna
    -------
    pandas.DataFrame con columnas:
        punto_acceso_origen | nombre_parada_origen |
        punto_acceso_destino | nombre_parada_destino |
        linea_a_usar | sentido | num_paradas_hasta_destino

    Notas
    -----
    - Cobertura: solo metro (02_*, A9_*) y cercanías (04_*).
      Los puntos de bus e interurbano quedan fuera.
    - Para cercanías, todos los puntos de la misma estación comparten el
      mismo código (04_L0_PY), por lo que pueden aparecer varias líneas
      posibles para el mismo par origen-destino. Eso es correcto y esperado.
    - Si un par origen-destino tiene múltiples líneas posibles, aparece
      una fila por cada opción.
    """
    if metro_pkl_path is None:
        metro_pkl_path = CHECK_DIR / "metro_multidigraph.pkl"
    if cercanias_pkl_path is None:
        cercanias_pkl_path = CHECK_DIR / "cercanias_multidigraph_sin_civis.pkl"
    if tabla_paradas_path is None:
        tabla_paradas_path = TOPO_DIR / "tabla_puntos_acceso_con_coords.csv"

    print("Cargando grafos de metro y cercanías...")
    with open(metro_pkl_path, "rb") as f:
        G_metro = pickle.load(f)
    with open(cercanias_pkl_path, "rb") as f:
        G_cercanias = pickle.load(f)

    print("Procesando rutas de metro...")
    rows_metro = _process_graph(G_metro)
    print(f"  → {len(rows_metro)} filas generadas para metro")

    print("Procesando rutas de cercanías...")
    rows_cercanias = _process_graph(G_cercanias)
    print(f"  → {len(rows_cercanias)} filas generadas para cercanías")

    df_rutas = pd.DataFrame(rows_metro + rows_cercanias).drop_duplicates()

    tabla_paradas = pd.read_csv(
        tabla_paradas_path,
        usecols=["punto_acceso", "nombre_parada"],
    ).drop_duplicates("punto_acceso")
    lookup = tabla_paradas.set_index("punto_acceso")["nombre_parada"].to_dict()

    df_rutas["nombre_parada_origen"] = df_rutas["punto_acceso_origen"].map(lookup)
    df_rutas["nombre_parada_destino"] = df_rutas["punto_acceso_destino"].map(lookup)

    df_rutas = df_rutas[[
        "punto_acceso_origen", "nombre_parada_origen",
        "punto_acceso_destino", "nombre_parada_destino",
        "linea_a_usar", "sentido", "num_paradas_hasta_destino",
    ]]

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_rutas.to_csv(out_path, index=False)
        print(f"CSV de rutas guardado en: {out_path} ({len(df_rutas):,} filas)")

    return df_rutas

def crear_matriz_transbordo_macro(
    x_pred_checked,
    transbordos_implicitos=None,
    top_n=20,
    out_dir=None,
    tabla_paradas_path=None,
):
    """
    Crea la matriz de TRANSBORDO agregada (nivel macro) en intervalos de 15 minutos,
    filtrada a los top_n intercambiadores con mayor volumen de transbordos total.

    Combina transbordos explícitos (filas 'TRANSBORDO' de x_pred_checked) e
    implícitos (self.transbordos_implicitos) en una única vista.

    Parámetros
    ----------
    x_pred_checked : pandas.DataFrame
        Output de SparkCheckTransbordo. Se usan las filas con tipo == 'TRANSBORDO'.
        Columnas necesarias:
          - punto_acceso      : estación donde se produce el transbordo (2º validación)
          - timestamp_validacion : momento del transbordo
    transbordos_implicitos : pandas.DataFrame, optional
        Output de asignar_transbordo(). Columnas necesarias:
          - transbordo         : estación física del transbordo
          - timestamp_transbordo : timestamp estimado de llegada al transbordo
        Si es None, solo se usan los transbordos explícitos.
    top_n : int, optional
        Número de intercambiadores a incluir en la matriz (ordenados por volumen total).
        Por defecto 20.
    out_dir : Path or str, optional
        Directorio donde guardar el parquet. Si se proporciona, se crea la carpeta
        y se guarda matriz_transbordo_macro.parquet dentro.
    tabla_paradas_path : Path or str, optional
        Ruta a tabla_puntos_acceso_con_coords.csv. Por defecto usa la ruta estándar.

    Retorna
    -------
    pandas.DataFrame con columnas:
        intervalo | estacion_transbordo | nombre_estacion | num_transbordos
    """
    if tabla_paradas_path is None:
        tabla_paradas_path = TOPO_DIR / "tabla_puntos_acceso_con_coords.csv"

    explicitos = (
        x_pred_checked[x_pred_checked["tipo"] == "TRANSBORDO"]
        [["prediccion_descenso", "timestamp_validacion"]]
        .copy()
        .rename(columns={
            "prediccion_descenso": "estacion_transbordo",
            "timestamp_validacion": "timestamp",
        })
    )

    frames = [explicitos]

    if transbordos_implicitos is not None and len(transbordos_implicitos) > 0:
        implicitos = (
            transbordos_implicitos[["transbordo", "timestamp_transbordo"]]
            .copy()
            .rename(columns={
                "transbordo": "estacion_transbordo",
                "timestamp_transbordo": "timestamp",
            })
        )
        frames.append(implicitos)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["intervalo"] = df["timestamp"].dt.floor("15min")

    macro = (
        df.groupby(["estacion_transbordo", "intervalo"])
        .size()
        .reset_index(name="num_transbordos")
    )

    volumen_total = (
        macro.groupby("estacion_transbordo")["num_transbordos"]
        .sum()
        .nlargest(top_n)
        .index
    )
    macro = macro[macro["estacion_transbordo"].isin(volumen_total)].copy()

    tabla_paradas = pd.read_csv(
        tabla_paradas_path,
        usecols=["punto_acceso", "nombre_parada"],
    ).drop_duplicates("punto_acceso")
    lookup = tabla_paradas.set_index("punto_acceso")["nombre_parada"].to_dict()
    macro["nombre_estacion"] = macro["estacion_transbordo"].map(lookup)

    macro = macro[[
        "intervalo", "estacion_transbordo", "nombre_estacion", "num_transbordos",
    ]].sort_values(["intervalo", "num_transbordos"], ascending=[True, False])

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "matriz_transbordo_macro.csv"
        macro.to_csv(out_path, index=False)
        print(f"Matriz macro guardada en: {out_path} ({len(macro):,} filas, top {top_n} estaciones)")

    return macro

def _lookup_rutas(df_main, rutas, key_origen_col, key_destino_col, suffix):
    """
    Hace un left join de df_main con el CSV de rutas usando las columnas indicadas
    como claves origen y destino. Añade las columnas linea_{suffix} y sentido_{suffix}.

    Cuando un par (origen, destino) tiene múltiples líneas posibles (habitual en
    cercanías, donde varias líneas comparten estación), se elige UNA al azar para
    evitar inflar el número de filas del resultado.

    Parámetros
    ----------
    df_main : pandas.DataFrame
        DataFrame principal.
    rutas : pandas.DataFrame
        CSV de rutas con columnas punto_acceso_origen, punto_acceso_destino,
        linea_a_usar, sentido.
    key_origen_col : str
        Columna de df_main que actúa como origen para el join.
    key_destino_col : str
        Columna de df_main que actúa como destino para el join.
    suffix : str
        Sufijo para las columnas resultantes ('llegada' o 'salida').

    Retorna
    -------
    pandas.DataFrame con las columnas linea_{suffix} y sentido_{suffix} añadidas.
    Mismo número de filas que df_main (sin inflación).
    """
    rutas_sub = rutas[["punto_acceso_origen", "punto_acceso_destino", "linea_a_usar", "sentido"]].copy()

    df_main = df_main.copy()
    df_main["_idx"] = np.arange(len(df_main))

    merged = df_main.merge(
        rutas_sub.rename(columns={
            "punto_acceso_origen": key_origen_col,
            "punto_acceso_destino": key_destino_col,
            "linea_a_usar": f"linea_{suffix}",
            "sentido": f"sentido_{suffix}",
        }),
        on=[key_origen_col, key_destino_col],
        how="left",
    )

    merged = merged.sample(frac=1).drop_duplicates(subset=["_idx"]).sort_values("_idx")
    merged = merged.drop(columns=["_idx"]).reset_index(drop=True)
    return merged

def _build_bus_lookup(topo_path):
    """
    Carga un fichero de topología de bus y devuelve un dict
    {DPAYPOINT: (linea, sentido)} usando NULINUSER como línea.
    """
    topo = pd.read_csv(topo_path, usecols=["DPAYPOINT", "NULINUSER", "sentido"])
    lookup = {}
    for _, row in topo.iterrows():
        if pd.notna(row["DPAYPOINT"]):
            lookup[row["DPAYPOINT"]] = (
                str(row["NULINUSER"]) if pd.notna(row["NULINUSER"]) else None,
                row["sentido"] if pd.notna(row["sentido"]) else None,
            )
    return lookup

def _apply_transport_lookup(df, orig_col, dest_col, suffix,
                             rutas, emt_lookup, inter_lookup):
    """
    Para cada fila de df determina el tipo de transporte a partir del prefijo
    de orig_col y aplica el lookup correspondiente:

      - Metro (02_, A9_) / Cercanías (04_): rutas_transbordo.csv (sin fallback)
      - Bus EMT (03_): topologia_solo_EMT.csv por DPAYPOINT (sin fallback)
      - Bus interurbano (resto): topologia_sin_EMT.csv por DPAYPOINT (sin fallback)

    Añade las columnas linea_{suffix} y sentido_{suffix} al dataframe.
    Si el par (origen, destino) no existe en rutas_transbordo.csv, ambas columnas
    quedan null — refleja predicciones de bajada inconsistentes con la red.
    """
    df = df.copy()

    mask_mc   = df[orig_col].str.startswith(_METRO_CERC_PREFIXES, na=False)
    mask_emt  = df[orig_col].str.startswith(_EMT_PREFIX, na=False)
    mask_inter = ~mask_mc & ~mask_emt & df[orig_col].notna()

    df[f"linea_{suffix}"] = None
    df[f"sentido_{suffix}"] = None

    if mask_mc.any():
        cols_a_quitar = [c for c in [f"linea_{suffix}", f"sentido_{suffix}"] if c in df.columns]
        mc_input = df[mask_mc].drop(columns=cols_a_quitar).reset_index(drop=True)
        mc = _lookup_rutas(mc_input, rutas, orig_col, dest_col, suffix)
        df.loc[mask_mc, f"linea_{suffix}"]   = mc[f"linea_{suffix}"].values
        df.loc[mask_mc, f"sentido_{suffix}"] = mc[f"sentido_{suffix}"].values

        mask_metro     = mask_mc & df[orig_col].str.startswith(("02_", "A9_"), na=False)
        mask_cercanias = mask_mc & df[orig_col].str.startswith("04_", na=False)
        null_linea = df[f"linea_{suffix}"].isna()
        df.loc[mask_metro     & null_linea, f"linea_{suffix}"] = "metro"
        df.loc[mask_cercanias & null_linea, f"linea_{suffix}"] = "cercanias"

    if mask_emt.any():
        df.loc[mask_emt, f"linea_{suffix}"]   = df.loc[mask_emt, orig_col].map(
            lambda x: emt_lookup.get(x, (None, None))[0]
        )
        df.loc[mask_emt, f"sentido_{suffix}"] = df.loc[mask_emt, orig_col].map(
            lambda x: emt_lookup.get(x, (None, None))[1]
        )

    if mask_inter.any():
        df.loc[mask_inter, f"linea_{suffix}"]   = df.loc[mask_inter, orig_col].map(
            lambda x: inter_lookup.get(x, (None, None))[0]
        )
        df.loc[mask_inter, f"sentido_{suffix}"] = df.loc[mask_inter, orig_col].map(
            lambda x: inter_lookup.get(x, (None, None))[1]
        )

    return df

def crear_matriz_transbordo_micro(
    x_pred_checked,
    transbordos_implicitos=None,
    rutas_csv_path=None,
    out_dir=None,
    tabla_paradas_path=None,
):
    """
    Crea la matriz de TRANSBORDO detallada (nivel micro) en intervalos de 15 minutos,
    desagregando por línea de llegada, línea de salida y sentido de la línea de salida.

    Ejemplo de fila resultante:
      intervalo=08:00, estacion=Moncloa, linea_llegada=bus611,
      linea_salida=L6, sentido_salida=LAGUNA, num_transbordos=13

    Combina transbordos explícitos e implícitos. Para los explícitos, la línea de
    llegada se deriva de 'pac_anterior' (validación previa de la misma tarjeta);
    si 'pac_anterior' es null (primera validación conocida) linea_llegada quedará
    como null.

    Parámetros
    ----------
    x_pred_checked : pandas.DataFrame
        Output de SparkCheckTransbordo. Se usan las filas con tipo == 'TRANSBORDO'.
        Columnas necesarias:
          - punto_acceso         : estación del transbordo
          - prediccion_descenso  : destino final tras el transbordo
          - timestamp_validacion : momento del transbordo
          - pac_anterior          : punto_acceso de la validación previa (puede ser null)
          - pred_descenso_anterior: prediccion_descenso de esa validación previa (puede ser null)
    transbordos_implicitos : pandas.DataFrame, optional
        Output de asignar_transbordo(). Columnas necesarias:
          - origen_transbordo    : punto donde inició el viaje
          - transbordo           : estación del transbordo
          - destino_transbordo   : destino final
          - timestamp_transbordo : timestamp estimado del transbordo
        Si es None, solo se usan los transbordos explícitos.
    rutas_csv_path : Path or str, optional
        Ruta al CSV generado por crear_csv_rutas_transbordo().
        Si es None, se busca en check_transbordo/rutas_transbordo.csv.
    out_dir : Path or str, optional
        Directorio donde guardar el parquet. Si se proporciona, se guarda
        matriz_transbordo_micro.parquet dentro.
    tabla_paradas_path : Path or str, optional
        Ruta a tabla_puntos_acceso_con_coords.csv.

    Retorna
    -------
    pandas.DataFrame con columnas:
        intervalo | estacion_transbordo | nombre_estacion |
        linea_llegada | sentido_llegada | linea_salida | sentido_salida |
        num_transbordos
    """
    if rutas_csv_path is None:
        rutas_csv_path = CHECK_DIR / "rutas_transbordo.csv"
    if tabla_paradas_path is None:
        tabla_paradas_path = TOPO_DIR / "tabla_puntos_acceso_con_coords.csv"

    rutas = pd.read_csv(rutas_csv_path)
    emt_lookup   = _build_bus_lookup(EMT_TOPO_PATH)
    inter_lookup = _build_bus_lookup(INTER_TOPO_PATH)

    explicitos = (
        x_pred_checked[x_pred_checked["tipo"] == "TRANSBORDO"]
        [["punto_acceso", "prediccion_descenso", "timestamp_validacion",
          "punto_acceso_next", "pred_descenso_next"]]
        .copy()
        .rename(columns={
            "prediccion_descenso": "estacion_transbordo",
            "timestamp_validacion": "timestamp",
        })
    )

    explicitos = _apply_transport_lookup(
        explicitos, "punto_acceso", "estacion_transbordo", "llegada",
        rutas, emt_lookup, inter_lookup,
    )
    explicitos = _apply_transport_lookup(
        explicitos, "punto_acceso_next", "pred_descenso_next", "salida",
        rutas, emt_lookup, inter_lookup,
    )

    explicitos = explicitos[[
        "estacion_transbordo", "timestamp",
        "linea_llegada", "sentido_llegada",
        "linea_salida", "sentido_salida",
    ]]

    frames = [explicitos]

    if transbordos_implicitos is not None and len(transbordos_implicitos) > 0:
        implicitos = (
            transbordos_implicitos[
                ["origen_transbordo", "transbordo", "destino_transbordo", "timestamp_transbordo"]
            ]
            .copy()
            .rename(columns={
                "transbordo": "estacion_transbordo",
                "timestamp_transbordo": "timestamp",
            })
        )

        implicitos = _apply_transport_lookup(
            implicitos, "origen_transbordo", "estacion_transbordo", "llegada",
            rutas, emt_lookup, inter_lookup,
        )
        implicitos = _apply_transport_lookup(
            implicitos, "estacion_transbordo", "destino_transbordo", "salida",
            rutas, emt_lookup, inter_lookup,
        )

        implicitos = implicitos[[
            "estacion_transbordo", "timestamp",
            "linea_llegada", "sentido_llegada",
            "linea_salida", "sentido_salida",
        ]]
        frames.append(implicitos)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["intervalo"] = df["timestamp"].dt.floor("15min")

    micro = (
        df.groupby(
            ["intervalo", "estacion_transbordo", "linea_llegada", "sentido_llegada",
             "linea_salida", "sentido_salida"],
            dropna=False,
        )
        .size()
        .reset_index(name="num_transbordos")
    )

    tabla_paradas = pd.read_csv(
        tabla_paradas_path,
        usecols=["punto_acceso", "nombre_parada"],
    ).drop_duplicates("punto_acceso")
    lookup = tabla_paradas.set_index("punto_acceso")["nombre_parada"].to_dict()
    micro["nombre_estacion"] = micro["estacion_transbordo"].map(lookup)

    micro = micro[[
        "intervalo", "estacion_transbordo", "nombre_estacion",
        "linea_llegada", "sentido_llegada",
        "linea_salida", "sentido_salida",
        "num_transbordos",
    ]].sort_values(["intervalo", "estacion_transbordo", "num_transbordos"],
                   ascending=[True, True, False])

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "matriz_transbordo_micro.csv"
        micro.to_csv(out_path, index=False)
        print(f"Matriz micro guardada en: {out_path} ({len(micro):,} filas)")

    return micro
