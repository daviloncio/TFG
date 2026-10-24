import pandas as pd
from pathlib import Path
from reconstruir_viajes import reconstruir_od

BASE_DIR = Path(__file__).resolve().parent
TOPO_DIR = BASE_DIR.parent / "topologia"
ZT_DIR   = BASE_DIR.parent / "zt1259"

def _load_zt_lookup(zt_path=None):
    """
    Carga el fichero paypoint_to_zt1259.csv y devuelve un dict
    {punto_acceso: cd_zt1259 (int)} para el enriquecimiento de la matriz OD.
    """
    if zt_path is None:
        zt_path = ZT_DIR / "paypoint_to_zt1259.csv"
    zt = pd.read_csv(zt_path, usecols=["DPAYPOINT", "CD_ZT1259"])
    zt = zt.dropna(subset=["CD_ZT1259"])
    zt["CD_ZT1259"] = zt["CD_ZT1259"].astype(int)
    return zt.set_index("DPAYPOINT")["CD_ZT1259"].to_dict()

def crear_matriz_od(x_pred_checked, out_dir=None, zt_path=None):
    """
    Crea la matriz Origen-Destino en intervalos de 15 minutos a partir de
    los viajes completos reconstruidos desde x_pred_checked.

    La granularidad espacial es la zonificación ZT-1259 del CRTM: cada parada
    de origen y destino se mapea a su zona tarifaria numérica mediante el
    fichero paypoint_to_zt1259.csv. El resultado es una matriz OD agregada a
    nivel de zona, mucho más compacta que la matriz a nivel de punto de acceso
    y coherente con la zonificación oficial utilizada por el CRTM y en la
    literatura de referencia.

    Los viajes cuyo origen o destino no tienen zona asignada en el fichero de
    mapeo quedan excluidos (en la práctica cero para los modos 02_, 03_, 04_,
    A9_ y 90_, que tienen cobertura completa en el fichero).

    Parámetros
    ----------
    x_pred_checked : pandas.DataFrame
        Output de check_transbordo. Debe contener al menos:
        - id_tarjeta          : identificador de tarjeta
        - punto_acceso        : parada de subida
        - prediccion_descenso : parada de destino predicha
        - timestamp_validacion: timestamp de la subida
        - tipo                : 'BAJADA', 'TRANSBORDO' o 'UNLINKED'

    out_dir : Path or str, optional
        Directorio donde guardar el CSV resultante. Si se proporciona,
        se crea la carpeta y se guarda matriz_od_zt1259.csv dentro.

    zt_path : Path or str, optional
        Ruta a paypoint_to_zt1259.csv. Por defecto usa zt1259/paypoint_to_zt1259.csv
        relativo al directorio padre de cubo_izquierdo.

    Retorna
    -------
    pandas.DataFrame
        Long format con columnas:
        intervalo | zona_origen | zona_destino | num_viajes
    pandas.DataFrame
        Viajes UNLINKED (sin predicción de destino), para uso posterior.
    """
    zt_lookup = _load_zt_lookup(zt_path)

    viajes, df_unlinked = reconstruir_od(x_pred_checked)

    viajes = viajes[viajes["origen_punto_acceso"] != viajes["destino_punto_acceso"]].copy()

    viajes["zona_origen"]  = viajes["origen_punto_acceso"].map(zt_lookup)
    viajes["zona_destino"] = viajes["destino_punto_acceso"].map(zt_lookup)

    n_antes = len(viajes)
    viajes = viajes.dropna(subset=["zona_origen", "zona_destino"])
    n_drop = n_antes - len(viajes)
    if n_drop > 0:
        print(f"  [crear_matriz_od] {n_drop} viajes descartados por falta de zona ZT-1259 "
              f"({n_drop/n_antes*100:.1f}%)")

    viajes["zona_origen"]  = viajes["zona_origen"].astype(int)
    viajes["zona_destino"] = viajes["zona_destino"].astype(int)

    viajes["intervalo"] = pd.to_datetime(viajes["timestamp_inicio"]).dt.floor("15min")

    od = (
        viajes
        .groupby(["intervalo", "zona_origen", "zona_destino"])
        .size()
        .reset_index(name="num_viajes")
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "matriz_od_zt1259.csv"
        od.to_csv(out_path, index=False)
        print(f"Matriz OD (ZT-1259) guardada en: {out_path} ({len(od):,} filas)")

    return od, df_unlinked
