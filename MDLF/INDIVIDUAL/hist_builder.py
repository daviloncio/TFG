"""
hist_builder.py
===============
Construye y cachea x_pred_checked_hist_indiv_mdlf.parquet: historial de viajes
procesados de los usuarios MDLF, generado desde los TTP CSVs de cada período.

Estrategia híbrida para reconstruir los destinos:
  · Bajada real   — si la siguiente validación tras una subida es un código de bajada
                    (el usuario validó al salir), usamos ese punto_acceso como destino
                    exacto (tipo='BAJADA').  Aplica principalmente a metro y cercanías.
  · Trip chaining — para las subidas sin bajada real posterior, usamos
                    SparkPrediccionDescenso (la siguiente subida del usuario aproxima
                    la estación de salida) + SparkCheckTransbordo para distinguir
                    BAJADA / TRANSBORDO / UNLINKED.

Flujo de build():
  1. Cargar usuarios con cercanías UNLINKED desde el almacén (unknown.parquet)
  2. Pre-filtrar TTP CSVs con pandas chunked → ttp_cerc_unlinked_users.parquet
     (solo filas de los ~6k usuarios; evita pasar 81 GB a Spark)
  3. Spark lee el parquet pequeño
  4. _preprocesar_hist(): filtro fecha, join coords, dedup Haversine,
     día operativo, filtro tipo_transporte, Window lead() → next_id_resultado /
     next_punto_acceso, extraer subidas
  5. Separar: subidas con bajada real vs subidas para predicción
  6. Subidas con bajada real → pandas, prediccion_descenso = next_punto_acceso, tipo='BAJADA'
  7. Subidas para predicción → SparkPrediccionDescenso → SparkCheckTransbordo
  8. Concatenar ambos grupos → guardar parquet

Columnas mínimas de salida requeridas por IndividualFeatureExtractor:
  id_tarjeta, timestamp_validacion, punto_acceso,
  prediccion_descenso, tipo_transporte, tipo
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MDLF_DIR     = Path(__file__).resolve().parent.parent   # MDLF/
_ROOT         = _MDLF_DIR.parent                          # david/estudios/
_CUBO_DIR     = _ROOT / "cubo_izquierdo"
_MDLF_ALMACEN = _MDLF_DIR / "almacen"
_TOPO_CSV     = _ROOT / "topologia" / "tabla_puntos_acceso_con_coords.csv"

# cubo_izquierdo en sys.path para acceder a spark_session, zona_tarifaria, etc.
if str(_CUBO_DIR) not in sys.path:
    sys.path.insert(0, str(_CUBO_DIR))

_DEFAULT_TTP_PATHS = [
    str(_ROOT / "descomprimido" / f"TTP_PERIODO_0{i}_2025.csv")
    for i in range(1, 7)
]

_TTP_COLS = [
    "id_tarjeta", "cod_em_operadora", "timestamp_validacion",
    "cod_grupo_validacion", "id_resultado", "id_tipo_titulo",
    "cod_perfil", "cod_descuento", "punto_acceso",
]

# id_resultado de subida (entrada al sistema)
_SUBIDA_CODES = [1, 2, 3, 4, 6, 11, 12, 63, 64, 65, 66]

# id_resultado de bajada real (salida validada del sistema: metro, cercanías)
# Cuando la siguiente validación de un usuario tiene uno de estos códigos,
# sabemos el destino exacto del viaje sin necesidad de trip chaining.
_BAJADA_CODES = [9, 44, 45, 81, 79, 34, 73, 75, 77, 10]


# ---------------------------------------------------------------------------
# Preprocesado
# ---------------------------------------------------------------------------

def _preprocesar_hist(df_spark, ts_max, hiperparams: dict):
    """
    Aplica filtros y preprocesado sobre el Spark DF ya pre-filtrado por usuarios.

    Parámetros
    ----------
    df_spark   : Spark DataFrame con columnas _TTP_COLS y timestamp_validacion
                 ya parseado como TimestampType (viene del parquet pre-filtrado).
    ts_max     : pd.Timestamp — límite superior del filtro de fecha.
    hiperparams: dict con al menos 'dia_virtual' (bool).

    Devuelve
    --------
    Spark DataFrame de subidas con columnas adicionales:
      next_id_resultado : id_resultado de la siguiente validación del usuario en el día
      next_punto_acceso : punto_acceso de esa siguiente validación
    Permite distinguir en build() qué subidas tienen bajada real asociada.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from spark_session import spark as _spark

    df = df_spark

    # 1. Filtro grueso de fecha
    ts_max_str = pd.Timestamp(ts_max).strftime("%Y-%m-%d %H:%M:%S")
    df = df.filter(F.col("timestamp_validacion") <= F.lit(ts_max_str).cast("timestamp"))

    # 2. Join con coords (LATITUD, LONGITUD, tipo_transporte, nombre_parada)
    df_stops_pd = (
        pd.read_csv(str(_TOPO_CSV))
        .drop_duplicates(subset=["punto_acceso"])
    )
    df_stops = (
        _spark.createDataFrame(df_stops_pd)
        .withColumn("LATITUD",  F.col("LATITUD").cast("double"))
        .withColumn("LONGITUD", F.col("LONGITUD").cast("double"))
    )
    df = df.join(df_stops, on="punto_acceso", how="left")


    # 3. Día operativo
    hora_corte = 4
    df = (
        df
        .withColumn("fecha_natural", F.to_date("timestamp_validacion"))
        .withColumn(
            "fecha_operativa",
            F.to_date(F.expr(f"timestamp_validacion - INTERVAL {hora_corte} HOURS"))
        )
    )
    fecha_col = "fecha_operativa" if hiperparams.get("dia_virtual", False) else "fecha_natural"
    df = df.withColumn("fecha", F.col(fecha_col))

    # Recortar días operativos incompletos en los bordes del dataset
    limites = df.select(
        F.min(F.to_date("timestamp_validacion")).alias("min_natural"),
        F.max(F.to_date("timestamp_validacion")).alias("max_natural"),
    ).first()
    df = df.filter(
        (F.col("fecha") >= F.lit(limites["min_natural"])) &
        (F.col("fecha") <= F.lit(limites["max_natural"]))
    )

    # 4. Deduplicación Haversine
    w = Window.partitionBy("id_tarjeta", "fecha").orderBy("timestamp_validacion")

    df = (
        df
        .withColumn("prev_timestamp",        F.lag("timestamp_validacion").over(w))
        .withColumn("prev_cod_em_operadora",  F.lag("cod_em_operadora").over(w))
        .withColumn("prev_lat",               F.lag("LATITUD").over(w))
        .withColumn("prev_lon",               F.lag("LONGITUD").over(w))
    )

    df = df.withColumn(
        "diff_seg",
        F.unix_timestamp("timestamp_validacion") - F.unix_timestamp("prev_timestamp"),
    )

    # Distancia Haversine (metros)
    _R        = 6_371_000.0
    _pi_180   = 3.141592653589793 / 180.0
    lat1 = F.col("prev_lat")  * F.lit(_pi_180)
    lon1 = F.col("prev_lon")  * F.lit(_pi_180)
    lat2 = F.col("LATITUD")   * F.lit(_pi_180)
    lon2 = F.col("LONGITUD")  * F.lit(_pi_180)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    _a   = F.sin(dlat / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin(dlon / 2) ** 2
    _c   = 2 * F.atan2(F.sqrt(_a), F.sqrt(1 - _a))
    df   = df.withColumn("dist_prev_m", F.lit(_R) * _c)

    casi_mismo_tiempo = F.abs(F.col("diff_seg")) <= 60
    mismo_modo        = F.col("cod_em_operadora") == F.col("prev_cod_em_operadora")
    misma_estacion    = F.col("dist_prev_m") < 200

    df = (
        df
        .withColumn("es_duplicado",
            F.when(casi_mismo_tiempo & mismo_modo & misma_estacion,  True).otherwise(False))
        .withColumn("caso_raro",
            F.when(casi_mismo_tiempo & mismo_modo & ~misma_estacion, True).otherwise(False))
    )

    # Eliminar duplicados exactos
    df = df.filter(~F.col("es_duplicado"))

    # Eliminar grupos (id_tarjeta, fecha) que contienen algún caso raro.
    # bad_groups es pequeño — se cachea para que el left_anti no recompute su
    # linaje (que ya incluye el lag Haversine) dos veces en el DAG.
    bad_groups = (
        df.filter(F.col("caso_raro"))
        .select("id_tarjeta", "fecha")
        .distinct()
        .cache()
    )
    bad_groups.count()   # materializar la caché antes del join
    df = df.join(bad_groups, on=["id_tarjeta", "fecha"], how="left_anti")
    # Checkpoint tras el left_anti: rompe el diamond DAG (df→bad_groups→join→df)
    # y evita que Spark regenere el linaje Haversine+lag en cada acción posterior.
    df = df.localCheckpoint(eager=True)

    cols_base = [
        "id_tarjeta", "id_resultado", "cod_em_operadora",
        "timestamp_validacion", "fecha", "tipo_transporte",
        "cod_perfil", "id_tipo_titulo", "nombre_parada",
        "punto_acceso", "LATITUD", "LONGITUD",
    ]
    df = df.select(cols_base)

    # 5. Filtro tipo_transporte IS NOT NULL
    #    IMPORTANTE: aplicar ANTES de calcular el lead() para que next_* apunte
    #    a la siguiente validación con estación conocida (donde normalmente estará
    #    la bajada real de metro/cercanías).
    df = df.filter(F.col("tipo_transporte").isNotNull())

    # 6. Window lead(): siguiente validación del usuario en el mismo día operativo.
    #    Se calcula sobre TODAS las validaciones (subidas + bajadas) para detectar
    #    cuándo la siguiente fila es una bajada real del sistema.
    #    Incluimos tipo_transporte y timestamp de la siguiente validación para poder
    #    aplicar en build() los mismos checks que crear_mapeos.py:
    #      · mismo modo de transporte
    #      · tiempo de viaje verosímil (cercanías ≤ 2.5h, metro/tranvía ≤ 1.25h)
    w2 = Window.partitionBy("id_tarjeta", "fecha").orderBy("timestamp_validacion")
    df = (
        df
        .withColumn("next_id_resultado",   F.lead("id_resultado").over(w2))
        .withColumn("next_punto_acceso",    F.lead("punto_acceso").over(w2))
        .withColumn("next_tipo_transporte", F.lead("tipo_transporte").over(w2))
        .withColumn("next_timestamp",       F.lead("timestamp_validacion").over(w2))
    )

    # 7. Extraer subidas (mantenemos next_* para que build() pueda hacer la separación)
    df = df.filter(F.col("id_resultado").cast("int").isin(_SUBIDA_CODES))

    return df


# ---------------------------------------------------------------------------
# Clase pública
# ---------------------------------------------------------------------------

class IndividualHistBuilder:
    """
    Construye y cachea el historial de viajes de los usuarios MDLF.

    Si output_path ya existe, load() lo devuelve directamente sin tocar Spark.
    Si no existe, build() lo genera desde los TTP CSVs con estrategia híbrida:
      · Bajadas reales (validación de salida) → destino exacto, tipo='BAJADA'
      · Subidas sin bajada real → trip chaining vía SparkPrediccionDescenso

    Parameters
    ----------
    nombre_almacen : str
        Nombre de la subcarpeta en MDLF/almacen/ (p.ej. '50mil_tarjetas_3_al_5_marzo').
    ttp_paths : list[str]
        Rutas a los TTP CSVs (6 períodos). Default: ../descomprimido/TTP_PERIODO_0X_2025.csv.
    hiperparams : dict
        Hiperparámetros del pipeline cubo izquierdo (dia_virtual, radio_busqueda, ...).
    almacen_dir : Path
        Raíz del almacén MDLF. Default: MDLF/almacen/.
    """

    def __init__(
        self,
        nombre_almacen : str,
        ttp_paths       : list[str] | None = None,
        hiperparams     : dict | None = None,
        almacen_dir     : Path = _MDLF_ALMACEN,
    ) -> None:
        self.nombre_almacen = nombre_almacen
        self.ttp_paths      = ttp_paths if ttp_paths is not None else _DEFAULT_TTP_PATHS
        self.hiperparams    = hiperparams or {}
        self._almacen_dir   = Path(almacen_dir)

    @property
    def output_path(self) -> Path:
        return self._almacen_dir / self.nombre_almacen / "x_pred_checked_hist_indiv_mdlf.parquet"

    @property
    def prefiltered_path(self) -> Path:
        return self._almacen_dir / self.nombre_almacen / "ttp_cerc_unlinked_users.parquet"

    def is_cached(self) -> bool:
        return self.output_path.exists()

    def load(self) -> pd.DataFrame:
        """Carga el historial ya generado desde el almacén."""
        return pd.read_parquet(self.output_path)

    def _prefilter_ttp(self, id_tarjetas: set) -> None:
        """
        Lee los TTP CSVs en chunks de pandas, filtra solo las filas de los
        usuarios con cercanías UNLINKED y guarda el resultado en prefiltered_path.

        Escritura incremental con pyarrow.ParquetWriter: cada chunk filtrado se
        vuelca directamente a disco sin acumular nada en memoria. Evita el pico
        de RAM del pd.concat final que duplicaba el tamaño del dataset filtrado.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        total_filas = 0
        writer = None

        for i, ttp_path in enumerate(self.ttp_paths, 1):
            filas_csv = 0
            print(f"      [HistBuilder] Pre-filtrando CSV {i}/{len(self.ttp_paths)}: {Path(ttp_path).name} ...")
            for chunk in pd.read_csv(
                ttp_path,
                sep="#",
                header=None,
                names=_TTP_COLS,
                encoding="UTF-8",
                chunksize=500_000,
                dtype=str,
            ):
                chunk_filt = chunk[chunk["id_tarjeta"].isin(id_tarjetas)].copy()
                if not chunk_filt.empty:
                    # Convertir timestamp antes de crear la tabla Arrow para que
                    # Spark no rechace TIMESTAMP(NANOS)
                    chunk_filt["timestamp_validacion"] = pd.to_datetime(
                        chunk_filt["timestamp_validacion"],
                        format="%d/%m/%Y %H:%M:%S",
                        errors="coerce",
                    ).astype("datetime64[us]")
                    table = pa.Table.from_pandas(chunk_filt, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(str(self.prefiltered_path), table.schema)
                    writer.write_table(table)
                    filas_csv += len(chunk_filt)

            total_filas += filas_csv
            print(f"        → {filas_csv:,} filas (total acumulado: {total_filas:,})")

        if writer is not None:
            writer.close()

        if total_filas == 0:
            raise ValueError(
                "Pre-filtrado TTP: 0 filas encontradas para los usuarios con cercanías UNLINKED. "
                "Verifica que los TTP CSVs corresponden al mismo conjunto de usuarios del almacén."
            )

        print(f"      [HistBuilder] ttp_cerc_unlinked_users.parquet guardado: {total_filas:,} filas")

    def prefilter(self) -> Path:
        """
        Ejecuta solo la fase pandas de pre-filtrado TTP (sin arrancar Spark).

        Genera ``ttp_cerc_unlinked_users.parquet`` si no existe ya en caché.
        Pensado para separar esta fase de la fase Spark: se puede ejecutar en un
        kernel, reiniciar, y luego llamar a ``build()`` en un kernel limpio para
        que Spark no compita con la RAM del pre-filtrado.

        Returns
        -------
        Path
            Ruta al parquet generado (o ya existente en caché).
        """
        _input_dir = self._almacen_dir / self.nombre_almacen
        if not _input_dir.exists():
            raise FileNotFoundError(
                f"No se encuentra el almacén '{self.nombre_almacen}' en "
                f"{self._almacen_dir}. Ejecuta primero el pipeline con hipótesis 21."
            )

        if self.prefiltered_path.exists():
            print(f"      [HistBuilder] ttp_cerc_unlinked_users.parquet ya en caché.")
            return self.prefiltered_path

        unknown = pd.read_parquet(_input_dir / "unknown.parquet")
        unknown_cerc = unknown[unknown["tipo_transporte"] == "cercanias"]
        id_tarjetas  = set(unknown_cerc["id_tarjeta"].astype(str))
        ts_max = pd.to_datetime(unknown["timestamp_validacion"]).max()
        print(f"      [HistBuilder] {len(id_tarjetas):,} usuarios con cercanías UNLINKED | ts_max = {ts_max}")

        print("      [HistBuilder] Pre-filtrando TTP CSVs con pandas chunked...")
        self._prefilter_ttp(id_tarjetas)
        return self.prefiltered_path

    def build(self) -> pd.DataFrame:
        """
        Genera el historial desde cero con estrategia híbrida:
          · Subidas con bajada real   → destino exacto (tipo='BAJADA')
          · Subidas sin bajada real   → SparkPrediccionDescenso + SparkCheckTransbordo
        Concatena ambos grupos y guarda el parquet en el almacén.
        """
        from spark_session import spark as _spark
        from pyspark.sql import functions as F
        from prediccion_descenso.spark_PrediccionDescenso import SparkPrediccionDescenso
        from check_transbordo.spark_check_transbordo import SparkCheckTransbordo

        _input_dir = self._almacen_dir / self.nombre_almacen
        if not _input_dir.exists():
            raise FileNotFoundError(
                f"No se encuentra el almacén '{self.nombre_almacen}' en "
                f"{self._almacen_dir}. Ejecuta primero el pipeline con hipótesis 21."
            )

        # ── 1. Usuarios con cercanías UNLINKED y límite de fecha ────────────
        unknown = pd.read_parquet(_input_dir / "unknown.parquet")

        # Solo necesitamos historial de usuarios con cercanías UNLINKED:
        # son los únicos que entran en PipelineMDLF (_build_labeled_set filtra cercanías).
        # Filtrar aquí reduce drásticamente el trabajo de Spark (~7x menos usuarios).
        unknown_cerc = unknown[unknown["tipo_transporte"] == "cercanias"]
        id_tarjetas  = set(unknown_cerc["id_tarjeta"].astype(str))

        ts_max = pd.to_datetime(unknown["timestamp_validacion"]).max()
        print(f"      [HistBuilder] {len(id_tarjetas):,} usuarios con cercanías UNLINKED | ts_max = {ts_max}")

        # ── 2. Pre-filtrar TTP CSVs con pandas (evita pasar 81 GB a Spark) ──
        if self.prefiltered_path.exists():
            print("      [HistBuilder] Reutilizando ttp_cerc_unlinked_users.parquet en caché...")
        else:
            print("      [HistBuilder] Pre-filtrando TTP CSVs con pandas chunked...")
            self._prefilter_ttp(id_tarjetas)

        # ── 3. Spark lee el parquet pequeño ────────────────────────────────
        print("      [HistBuilder] Spark leyendo parquet pre-filtrado...")
        df_raw = _spark.read.parquet(str(self.prefiltered_path))

        # ── 4. Preprocesado ─────────────────────────────────────────────────
        print("      [HistBuilder] Preprocesando (fecha, dedup, lead bajadas)...")
        subidas = _preprocesar_hist(df_raw, ts_max, self.hiperparams)
        subidas = subidas.localCheckpoint(eager=True)
        n_total = subidas.count()
        print(f"      [HistBuilder] Subidas totales tras preprocesado: {n_total:,}")

        if n_total == 0:
            raise ValueError(
                "El historial tiene 0 subidas tras el preprocesado. "
                "Verifica que los TTP CSVs y el almacén corresponden al mismo período."
            )

        # ── 5. Separar por tipo de destino ──────────────────────────────────
        #  · con_bajada_real : la siguiente validación del usuario es una salida
        #                      validada (metro/cercanías) → destino exacto conocido
        #  · para_prediccion : no hay bajada real → trip chaining necesario
        #
        # Replicamos la lógica de crear_mapeos.py adaptada a Spark:
        #   1. next_id_resultado ∈ codigos_descenso
        #   2. mismo tipo de transporte (solo metro/cercanías/tranvía tienen torniquete salida)
        #   3. tiempo de viaje verosímil: cercanías ≤ 2.5h, metro/tranvía ≤ 1.25h
        _SISTEMAS_CON_TORNIQUETE = ["cercanias", "metro", "metro_ligero_tranvia"]

        diff_seg_next = (
            F.unix_timestamp("next_timestamp") - F.unix_timestamp("timestamp_validacion")
        )
        tiempo_plausible = (
            F.when(
                F.col("tipo_transporte") == "cercanias",
                diff_seg_next <= 9_000,     # 2h 30min
            ).when(
                F.col("tipo_transporte").isin("metro", "metro_ligero_tranvia"),
                diff_seg_next <= 4_500,     # 1h 15min
            ).otherwise(F.lit(False))
        )

        tiene_bajada_real = (
            F.col("next_id_resultado").cast("int").isin(_BAJADA_CODES)          # código de salida
            & F.col("tipo_transporte").isin(_SISTEMAS_CON_TORNIQUETE)            # sistema con torniquete
            & (F.col("tipo_transporte") == F.col("next_tipo_transporte"))        # mismo modo
            & tiempo_plausible                                                    # tiempo verosímil
        )

        subidas_con_bajada = subidas.filter(tiene_bajada_real)
        subidas_para_pred  = subidas.filter(~tiene_bajada_real)

        n_real = subidas_con_bajada.count()
        n_pred = n_total - n_real
        print(f"      [HistBuilder] Con bajada real (destino exacto): {n_real:,} "
              f"({100*n_real/n_total:.1f}%)")
        print(f"      [HistBuilder] Para trip chaining (SparkPrediccionDescenso): {n_pred:,} "
              f"({100*n_pred/n_total:.1f}%)")

        # ── 6. Subidas con bajada real → pandas directo ─────────────────────
        _cols_base = [
            "id_tarjeta", "id_resultado", "cod_em_operadora",
            "timestamp_validacion", "fecha", "tipo_transporte",
            "cod_perfil", "id_tipo_titulo", "nombre_parada",
            "punto_acceso", "LATITUD", "LONGITUD",
            "next_punto_acceso",
        ]
        real_trips_pd = (
            subidas_con_bajada
            .select(_cols_base)
            .toPandas()
            .rename(columns={"next_punto_acceso": "prediccion_descenso"})
        )
        real_trips_pd["tipo"] = "BAJADA"

        # ── 7. Viajes sin bajada real → SparkPrediccionDescenso ─────────────
        if n_pred > 0:
            # Eliminar columnas next_* antes de pasar a SparkPrediccionDescenso
            subidas_para_pred = subidas_para_pred.drop(
                "next_id_resultado", "next_punto_acceso",
                "next_tipo_transporte", "next_timestamp",
            )
            subidas_para_pred = subidas_para_pred.localCheckpoint(eager=True)

            print("      [HistBuilder] SparkPrediccionDescenso (trip chaining)...")
            instancia = SparkPrediccionDescenso(
                subidas_para_pred,
                hipotesis={5},   # Spark version; check_transbordo se llama directamente
                hiperparams=self.hiperparams,
            )
            instancia.lote_inicial = instancia.lote_inicial.localCheckpoint(eager=True)
            if self.hiperparams.get("paper_original") in ["munizaga", "lee"]:
                instancia.construir_lotes_next_day()
            else:
                instancia.construir_lotes()
            instancia.lote_primera_sub = instancia.lote_primera_sub.localCheckpoint(eager=True)
            instancia.lote_segunda_sub = instancia.lote_segunda_sub.localCheckpoint(eager=True)
            # zonas_permitidas no viene de los TTP CSVs — se añade con todas las zonas
            # para que SparkPrediccionDescenso no falle (no filtra candidatos por corona)
            _todas_zonas = F.array(*[F.lit(z) for z in ["A", "B1", "B2", "B3", "C1", "C2", "E1", "E2"]])
            instancia.lote_segunda_sub = instancia.lote_segunda_sub.withColumn("zonas_permitidas", _todas_zonas)
            x_1sub, x_2sub, y_pred, _ = instancia.spark_predicciones()

            # Asignar prediccion_descenso antes de check_transbordo
            x_1sub["prediccion_descenso"] = y_pred.to_numpy()

            print("      [HistBuilder] SparkCheckTransbordo...")
            checker = SparkCheckTransbordo()
            pred_trips_pd = checker.is_transbordo(x_1sub, x_2sub)
            # Excluir UNLINKED: no hay destino conocido → no aportan al historial
            pred_trips_pd = pred_trips_pd[pred_trips_pd["tipo"] != "UNLINKED"]
        else:
            pred_trips_pd = pd.DataFrame()

        # ── 8. Concatenar bajadas reales + viajes por predicción ────────────
        partes = [df for df in [real_trips_pd, pred_trips_pd] if len(df) > 0]
        x_pred_checked_hist = pd.concat(partes, ignore_index=True)

        tipos = x_pred_checked_hist["tipo"].value_counts().to_dict()
        print(f"      [HistBuilder] Historial final: {len(x_pred_checked_hist):,} viajes | {tipos}")

        # ── 9. Guardar ───────────────────────────────────────────────────────
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        x_pred_checked_hist.to_parquet(self.output_path, index=False)
        print(f"      [HistBuilder] Guardado: {self.output_path.name}")

        return x_pred_checked_hist
