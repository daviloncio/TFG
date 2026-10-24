"""
spark_crear_mapeos.py
Versión Spark de crear_mapeos. Misma firma que la versión pandas.
Usar cuando 5 in hipotesis (datasets grandes).

Lógica replicada fielmente del loop fila-a-fila original mediante:
  - Window partitionBy(id_tarjeta).orderBy(timestamp_validacion)
  - lag(1) para la fila anterior  |  lag(2) para el caso de validación doble
  - Self-join para recuperar las filas completas de x e y
"""
from __future__ import annotations

from pyspark.sql import functions as F, Window
from pyspark.sql import SparkSession
from spark_session import spark
import pandas as pd

CODIGOS_DESCENSO = frozenset({'9', '44', '45', '81', '79', '34', '73', '75', '77', '10'})

_PREFIJO_MEDIO: dict[str, str] = {
    "02": "metro",               "A9": "metro",
    "03": "bus_emt",             "04": "cercanias",
    "AD": "metro_ligero_tranvia","AF": "metro_ligero_tranvia","B1": "metro_ligero_tranvia",
    "1A": "bus_interurbano",     "1B": "bus_interurbano",     "20": "bus_interurbano",
    "23": "bus_interurbano",     "24": "bus_interurbano",     "25": "bus_interurbano",
    "26": "bus_interurbano",     "2C": "bus_interurbano",     "2E": "bus_interurbano",
    "2F": "bus_interurbano",     "30": "bus_interurbano",     "31": "bus_interurbano",
    "33": "bus_interurbano",     "34": "bus_interurbano",     "35": "bus_interurbano",
    "38": "bus_interurbano",     "39": "bus_interurbano",     "3A": "bus_interurbano",
    "3B": "bus_interurbano",     "3C": "bus_interurbano",     "3D": "bus_interurbano",
    "43": "bus_interurbano",     "45": "bus_interurbano",     "47": "bus_interurbano",
    "4A": "bus_interurbano",     "90": "bus_interurbano",
}

def spark_crear_mapeos(
    validaciones: pd.DataFrame,
    hiperparams: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (x_test_df, y_test_df) con las mismas filas y columnas que
    crear_mapeos(), pero usando Spark internamente para escalar a millones de filas.

    Args:
        validaciones: DataFrame pandas ordenado por (id_tarjeta, timestamp_validacion).
        hiperparams:  Mismo diccionario de hiperparámetros del pipeline.

    Returns:
        x_test_df, y_test_df  (pandas DataFrames, mismo esquema que validaciones)
    """
    paper = hiperparams['paper_original']

    if validaciones.empty:
        return validaciones.copy(), validaciones.copy()

    sdf = spark.createDataFrame(validaciones.reset_index(drop=True))
    w   = Window.partitionBy("id_tarjeta").orderBy("timestamp_validacion")
    sdf = sdf.withColumn("_seq", F.row_number().over(w))

    medio_map = F.create_map([F.lit(x) for kv in _PREFIJO_MEDIO.items() for x in kv])
    sdf = sdf.withColumn("_medio", medio_map[F.substring("punto_acceso", 1, 2)])

    sdf = (sdf
        .withColumn("_p1_id_res", F.lag("id_resultado",         1).over(w))
        .withColumn("_p1_pac",    F.lag("punto_acceso",         1).over(w))
        .withColumn("_p1_ts",     F.lag("timestamp_validacion", 1).over(w))
        .withColumn("_p1_fecha",  F.lag("fecha",                1).over(w))
        .withColumn("_p1_medio",  F.lag("_medio",               1).over(w))
        .withColumn("_p2_ts",     F.lag("timestamp_validacion", 2).over(w))
        .withColumn("_p2_medio",  F.lag("_medio",               2).over(w))
    )

    is_bajada      = F.col("id_resultado").isin(*list(CODIGOS_DESCENSO))
    prev_is_subida = (
        F.col("_p1_id_res").isNotNull() &
        ~F.col("_p1_id_res").isin(*list(CODIGOS_DESCENSO))
    )

    valid_medio = F.col("_p1_medio").isin("metro", "cercanias", "metro_ligero_tranvia")

    diff_pac = (
        F.lit(True) if paper == 'cubo'
        else (F.col("punto_acceso") != F.col("_p1_pac"))
    )
    prereq = is_bajada & prev_is_subida & valid_medio & diff_pac

    mismo_medio = F.col("_medio") == F.col("_p1_medio")
    mismo_dia   = (
        F.lit(True) if paper == 'cubo'
        else (F.col("fecha") == F.col("_p1_fecha"))
    )
    diff_secs = F.unix_timestamp("timestamp_validacion") - F.unix_timestamp("_p1_ts")
    tiempo_ok = (
        ((F.col("_p1_medio") == "cercanias")                       & (diff_secs <= 9_000))
        | ((F.col("_p1_medio").isin("metro", "metro_ligero_tranvia")) & (diff_secs <= 4_500))
        | F.lit(paper == 'cubo')
    )
    main_cond = mismo_medio & mismo_dia & tiempo_ok

    main_pairs = (
        sdf.filter(prereq & main_cond)
           .select(
               "id_tarjeta",
               F.col("_seq").alias("y_seq"),
               (F.col("_seq") - 1).alias("x_seq"),
           )
    )

    if paper != 'cubo':
        diff_10s       = (F.abs(F.unix_timestamp("timestamp_validacion") - F.unix_timestamp("_p1_ts")) <= 10)
        mismo_dia_p1   = F.col("fecha") == F.col("_p1_fecha")
        mismo_medio_p2 = F.col("_medio") == F.col("_p2_medio")

        mismo_dia_p2   = F.col("fecha") == F.to_date(F.col("_p2_ts"))

        edge_pairs = (
            sdf.filter(
                prereq & ~main_cond &
                diff_10s & mismo_dia_p1 & mismo_medio_p2 & mismo_dia_p2
            )
            .select(
                "id_tarjeta",
                F.col("_seq").alias("y_seq"),
                (F.col("_seq") - 2).alias("x_seq"),
            )
        )
        all_pairs = main_pairs.union(edge_pairs)
    else:
        all_pairs = main_pairs

    orig_cols = list(validaciones.columns)
    sdf_slim  = sdf.select(
        ["id_tarjeta", "_seq"] + [c for c in orig_cols if c != "id_tarjeta"]
    )

    y_full = (
        all_pairs.select("id_tarjeta", "y_seq")
        .join(sdf_slim.withColumnRenamed("_seq", "y_seq"), on=["id_tarjeta", "y_seq"])
        .orderBy("id_tarjeta", "y_seq")
        .select(orig_cols)
    )

    x_full = (
        all_pairs.select("id_tarjeta", "y_seq", F.col("x_seq").alias("_x"))
        .join(sdf_slim.withColumnRenamed("_seq", "_x"), on=["id_tarjeta", "_x"])
        .orderBy("id_tarjeta", "y_seq")
        .select(orig_cols)
    )

    x_pd = x_full.toPandas()
    y_pd = y_full.toPandas()

    if paper != 'cubo':
        mask_bp = x_pd["nombre_parada"] == "BARRIO DEL PUERTO"
        x_pd = x_pd.loc[~mask_bp].reset_index(drop=True)
        y_pd = y_pd.loc[~mask_bp].reset_index(drop=True)

    y_pd.loc[y_pd["punto_acceso"] == "02_L12_P22", "punto_acceso"] = "02_L3_P58"

    return x_pd.reset_index(drop=True), y_pd.reset_index(drop=True)
