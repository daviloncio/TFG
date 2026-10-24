from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.window import Window
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spark_session import spark

BASE_DIR = Path(__file__).resolve().parent

class SparkCheckTransbordo:
    def __init__(self):
        self.spark = spark

        self.buffer = broadcast(
            spark.read.option("header", True)
            .csv(str(BASE_DIR / "tiempos_viaje_pares.csv"))
            .withColumn("tiempo_viaje", F.col("tipo").cast("double"))
            .drop("tipo")
        )

    def is_transbordo(self, lote_primera_sub, lote_segunda_sub):
        """
        Clasifica cada validación de lote_primera_sub como TRANSBORDO, BAJADA o UNLINKED,
        y añade columnas con datos de la validación siguiente del mismo id_tarjeta.

        Parámetros
        ----------
        lote_primera_sub : pandas.DataFrame o pyspark.sql.DataFrame
            Debe incluir: pair_id, id_tarjeta, timestamp_validacion,
                          punto_acceso, prediccion_descenso.
        lote_segunda_sub : pandas.DataFrame o pyspark.sql.DataFrame
            Debe incluir: pair_id, timestamp_validacion.

        Retorna
        -------
        pandas.DataFrame
            lote_primera_sub con las columnas añadidas:
            - 'tipo'              : TRANSBORDO / BAJADA / UNLINKED
            - 'punto_acceso_next' : punto_acceso de la siguiente validación del mismo id_tarjeta
                                    (null si no hay validación siguiente en el lote).
            - 'pred_descenso_next': prediccion_descenso de esa siguiente validación.
                                    Juntos permiten derivar linea_salida en transbordos explícitos.
        """
        TIEMPO_MTT = 1080

        import pandas as pd
        if isinstance(lote_primera_sub, pd.DataFrame):
            lote_primera_sub = spark.createDataFrame(lote_primera_sub)
        if isinstance(lote_segunda_sub, pd.DataFrame):
            lote_segunda_sub = spark.createDataFrame(lote_segunda_sub)

        w_tarjeta = Window.partitionBy("id_tarjeta").orderBy("timestamp_validacion")
        lote_primera_sub = (
            lote_primera_sub
            .withColumn("punto_acceso_next", F.lead("punto_acceso", 1).over(w_tarjeta))
            .withColumn("pred_descenso_next", F.lead("prediccion_descenso", 1).over(w_tarjeta))
        )

        ts2 = lote_segunda_sub.select(
            "pair_id",
            F.col("timestamp_validacion").alias("ts_2sub"),
        )

        resultado = (
            lote_primera_sub
            .join(ts2, on="pair_id", how="left")
            .withColumn(
                "diff_sec",
                F.unix_timestamp("ts_2sub") - F.unix_timestamp("timestamp_validacion"),
            )
            .withColumn(
                "id_par",
                F.concat(F.col("punto_acceso"), F.lit("_y_"), F.col("prediccion_descenso")),
            )
            .join(self.buffer, on="id_par", how="left")
            .withColumn(
                "tipo",
                F.when(
                    F.col("prediccion_descenso") == "unknown",
                    "UNLINKED",
                ).when(
                    (F.col("diff_sec") >= 0)
                    & (F.col("punto_acceso") != F.col("prediccion_descenso"))
                    & (F.col("diff_sec") <= F.col("tiempo_viaje") + TIEMPO_MTT),
                    "TRANSBORDO",
                ).otherwise("BAJADA"),
            )
            .withColumn(
                "timestamp_descenso",
                F.to_timestamp(F.unix_timestamp("timestamp_validacion") + F.col("tiempo_viaje")),
            )
            .drop("ts_2sub", "diff_sec", "id_par", "tiempo_viaje")
        )

        return resultado.toPandas()
