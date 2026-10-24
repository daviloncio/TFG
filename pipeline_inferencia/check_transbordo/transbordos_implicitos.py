from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PRED_DIR = BASE_DIR.parent / "prediccion_descenso"

def detectar_transbordos_implicitos(x1_sub):
    """
    Detecta transbordos implícitos en el conjunto de validaciones.

    Un transbordo implícito ocurre cuando el punto de acceso donde el viajero
    sube (punto_acceso) y el punto de acceso donde se predice que baja
    (prediccion_descenso) pertenecen a líneas distintas. Esto implica que el
    viajero tuvo que realizar un transbordo físico entre medios o líneas que
    no queda registrado explícitamente en los datos de validación.

    Para cada registro donde se detecta este caso, se construye un nuevo
    registro con los campos del tramo de transbordo. El resto de campos
    (p.ej. información del viajero, coordenadas, etc.) se completarán
    mediante funciones específicas que se llamarán posteriormente.

    Parámetros
    ----------
    x1_sub : pandas.DataFrame
        Output de prediccion_descenso. Debe contener al menos:
        - punto_acceso         : punto de acceso de subida
        - prediccion_descenso  : punto de acceso de bajada predicho
        - timestamp_validacion : timestamp de la subida
    Retorna
    -------
    pyspark.sql.DataFrame
        Registros de transbordos implícitos con los campos:
        - origen_transbordo   : punto_acceso de origen del transbordo
        - timestamp_subida    : timestamp de subida en el origen
        - destino_transbordo  : punto_acceso de destino del transbordo
    """
    from check_transbordo.spark_check_transbordo import spark

    corr = broadcast(
        spark.read.option("header", True)
        .csv(str(PRED_DIR / "correspondencias_lineas.csv"))
        .withColumn("lineas", F.from_json(F.col("lineas"), ArrayType(StringType())))
    )

    df = spark.createDataFrame(x1_sub)

    df = df.join(
        corr.withColumnRenamed("lineas", "lineas_origen"),
        on="punto_acceso",
        how="left",
    )

    df = df.join(
        corr.select(
            F.col("punto_acceso").alias("prediccion_descenso"),
            F.col("lineas").alias("lineas_destino"),
        ),
        on="prediccion_descenso",
        how="left",
    )

    df = df.withColumn(
        "es_transbordo_implicito",
        F.when(
            F.col("lineas_origen").isNull() | F.col("lineas_destino").isNull(),
            F.lit(False),
        ).otherwise(
            F.size(F.array_intersect("lineas_origen", "lineas_destino")) == 0
        ),
    )

    transbordos = df.filter(F.col("es_transbordo_implicito")).select(
        F.col("punto_acceso").alias("origen_transbordo"),
        F.col("timestamp_validacion").alias("timestamp_subida"),
        F.col("prediccion_descenso").alias("destino_transbordo"),
    )

    return transbordos

def asignar_transbordo(transbordos):
    """
    Asigna un punto de transbordo concreto y su timestamp estimado a cada
    registro de transbordo implícito.

    Para cada par (origen_transbordo, destino_transbordo), consulta
    transbordos_posibles.csv para obtener los candidatos. Si hay varios,
    elige uno aleatoriamente. A continuación busca en tiempos_viaje_pares.csv el
    tiempo estimado de viaje desde el origen hasta el transbordo elegido y
    calcula el timestamp en que se estima que el viajero llegó a dicha
    estación de transbordo.

    Parámetros
    ----------
    transbordos : pyspark.sql.DataFrame
        Output de detectar_transbordos_implicitos. Debe contener:
        - origen_transbordo  : punto de acceso de origen
        - destino_transbordo : punto de acceso de destino final
        - timestamp_subida   : timestamp de subida en el origen
        - timestamp_descenso : timestamp estimado de bajada en el destino

    Retorna
    -------
    pyspark.sql.DataFrame
        El DataFrame de entrada enriquecido con:
        - transbordo          : punto de acceso donde se realiza el transbordo
        - timestamp_transbordo: timestamp estimado de llegada al transbordo
    """
    from check_transbordo.spark_check_transbordo import spark

    posibles = broadcast(
        spark.read.option("header", True)
        .csv(str(BASE_DIR / "transbordos_posibles.csv"))
    )

    df = transbordos.join(
        posibles,
        on=[
            transbordos["origen_transbordo"] == posibles["origen"],
            transbordos["destino_transbordo"] == posibles["destino"],
        ],
        how="left",
    ).drop("origen", "destino")

    w = Window.partitionBy("origen_transbordo", "destino_transbordo", "timestamp_subida").orderBy(F.rand())
    df = (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    buffer = broadcast(
        spark.read.option("header", True)
        .csv(str(BASE_DIR / "tiempos_viaje_pares.csv"))
        .withColumn("tiempo_viaje", F.col("tipo").cast("double"))
        .drop("tipo")
    )

    df = (
        df.withColumn(
            "id_par",
            F.concat(F.col("origen_transbordo"), F.lit("_y_"), F.col("transbordo")),
        )
        .join(buffer, on="id_par", how="left")
        .drop("id_par")
    )

    df = df.withColumn(
        "timestamp_transbordo",
        F.to_timestamp(F.unix_timestamp("timestamp_subida") + F.col("tiempo_viaje")),
    ).drop("tiempo_viaje")

    df = df.filter(
        F.col("transbordo").isNotNull() & F.col("timestamp_transbordo").isNotNull()
    )

    return df
