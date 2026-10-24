import pandas as pd
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, trim
from pyspark.sql.functions import to_timestamp
from zona_tarifaria import *
from spark_session import spark

_BASE_DIR = Path(__file__).resolve().parent

def establecer_fecha_virutal_o_natural(df,hora_corte=5,hiperparams=None):
    """
    LA REGLA PARA ENTENDER CÓMO VAMOS A USAR LAS COLUMNAS ES: 
        si queremos agrupar -> fecha (puede ser natural o virtual)
        si queremos ordenar o medir tiempos-> timestamp validacion 
    """
    df = (
        df
        .withColumn("fecha_natural", F.to_date("timestamp_validacion"))
        .withColumn(
            "fecha_operativa",
            F.to_date(F.expr(f"timestamp_validacion - INTERVAL {hora_corte} HOURS"))
        )
    )

    fecha_col = "fecha_operativa" if hiperparams['dia_virtual'] else "fecha_natural"
    df = df.withColumn("fecha", F.col(fecha_col))
    return df

def recortar_dias_operativos_incompletos(df):
    limites = df.select(
        F.min(F.to_date("timestamp_validacion")).alias("min_natural"),
        F.max(F.to_date("timestamp_validacion")).alias("max_natural")
    ).first()

    min_natural = limites["min_natural"]
    max_natural = limites["max_natural"]

    return df.filter(
        (F.col("fecha") >= F.lit(min_natural)) &
        (F.col("fecha") <= F.lit(max_natural))
    )

def crear_dfs_y_descartar_duplicados(path,hiperparams):

    
    """Leemos el CSV, renombramos columnas y quitamos:
       - duplicados: misma tarjeta, mismo modo, |Δt| <= 60s, dist < 200m
       - casos raros: misma tarjeta, mismo modo, |Δt| <= 60s, dist >= 200m
    """

    
    df_spark = (spark.read
                .option("header", "false")
                .option("delimiter", "#")
                .option("encoding", "UTF-8")
                .csv(path))
    
    from pyspark.sql.functions import to_timestamp
    
    df_spark = df_spark.withColumn("timestamp", to_timestamp("_c2", "dd/MM/yyyy HH:mm:ss"))
    df_spark = df_spark.drop("_c2")
    
    columnas_ordenadas = ['_c0', '_c1', 'timestamp', '_c3', '_c4', '_c5', '_c6', '_c7', '_c8']
    df_spark = df_spark.select(*columnas_ordenadas)
    
    column_names = [
        'id_tarjeta',
        'cod_em_operadora',
        'timestamp_validacion',
        'cod_grupo_validacion',
        'id_resultado',
        'id_tipo_titulo',
        'cod_perfil',
        'cod_descuento',
        'punto_acceso'
    ]
    df_spark = df_spark.toDF(*column_names)
    print(f'numero de filas en el csv: {df_spark.count()}')

    df_stops_pd = pd.read_csv(_BASE_DIR.parent / "topologia" / "tabla_puntos_acceso_con_coords.csv")
    df_stops_pd = df_stops_pd.drop_duplicates(subset=["punto_acceso"])
    df_stops = spark.createDataFrame(df_stops_pd)
    
    df_stops = (
        df_stops
        .withColumn("LATITUD", F.col("LATITUD").cast("double"))
        .withColumn("LONGITUD", F.col("LONGITUD").cast("double"))
        .withColumn("tipo_transporte",F.col("tipo_transporte"))
    )
    

    df = df_spark.join(df_stops, on="punto_acceso", how="left")

    df = df.withColumn("cod_em_operadora", F.col("cod_em_operadora"))
    

    df = establecer_fecha_virutal_o_natural(df,4,hiperparams)
    df = recortar_dias_operativos_incompletos(df)

    w = Window.partitionBy("id_tarjeta", "fecha").orderBy("timestamp_validacion")
    
    df = (
        df
        .withColumn("prev_timestamp", F.lag("timestamp_validacion").over(w))
        .withColumn("prev_punto_acceso", F.lag("punto_acceso").over(w))
        .withColumn("prev_cod_em_operadora", F.lag("cod_em_operadora").over(w))
        .withColumn("prev_id_tipo_titulo", F.lag("id_tipo_titulo").over(w))
        .withColumn("prev_cod_perfil", F.lag("cod_perfil").over(w))
        .withColumn("prev_lat", F.lag("LATITUD").over(w))
        .withColumn("prev_lon", F.lag("LONGITUD").over(w))
        .withColumn("prev_tipo_transporte", F.lag("tipo_transporte").over(w))
        .withColumn("prev_nombre_parada", F.lag("nombre_parada").over(w))  
    )
    

    df = df.withColumn(
        "diff_seg",
        F.unix_timestamp("timestamp_validacion") - F.unix_timestamp("prev_timestamp")
    )
    

    R = 6371000.0
    deg2rad = lambda col: col * (F.lit(3.141592653589793) / F.lit(180.0))
    
    lat1 = deg2rad(F.col("prev_lat"))
    lon1 = deg2rad(F.col("prev_lon"))
    lat2 = deg2rad(F.col("LATITUD"))
    lon2 = deg2rad(F.col("LONGITUD"))
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = F.sin(dlat / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin(dlon / 2) ** 2
    c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
    dist_m = R * c
    
    df = df.withColumn("dist_prev_m", dist_m)
    

    casi_mismo_tiempo = (F.abs(F.col("diff_seg")) <= 60)
    mismo_modo = (F.col("cod_em_operadora") == F.col("prev_cod_em_operadora"))
    misma_estacion = (F.col("dist_prev_m") < 200)
    
    df = df.withColumn(
        "es_duplicado",
        F.when(casi_mismo_tiempo & mismo_modo & misma_estacion, F.lit(True)).otherwise(F.lit(False))
    )
    
    df = df.withColumn(
        "caso_raro_distinta_estacion",
        F.when(casi_mismo_tiempo & mismo_modo & (~misma_estacion), F.lit(True)).otherwise(F.lit(False))
    )

    df_raros = df.filter(F.col("caso_raro_distinta_estacion"))
    df_dup = df.filter(F.col("es_duplicado"))

    n_dup = df_dup.count()
    n_raros = df_raros.count()
    print("Duplicados:", n_dup)
    print("Casos raros (≤60s, mismo modo, >200m):", n_raros)

    raros_sample = (
        df_raros
        .orderBy(F.rand())
        .limit(20)
        .withColumn("pair_id", F.monotonically_increasing_id())
    )

    prev_rows = raros_sample.select(
        "pair_id",
        F.lit("prev").alias("tipo_fila"),
        "id_tarjeta",
        "fecha",
        F.col("prev_timestamp").alias("timestamp_validacion"),
        F.col("prev_cod_em_operadora").alias("cod_em_operadora"),
        F.col("prev_punto_acceso").alias("punto_acceso"),
        F.col("prev_nombre_parada").alias("nombre_parada"),
        F.col("prev_id_tipo_titulo").alias("id_tipo_titulo"),
        F.col("prev_cod_perfil").alias("cod_perfil"),
        F.col("prev_tipo_transporte").alias("tipo_transporte"),
        F.col("prev_lat").alias("LATITUD"),
        F.col("prev_lon").alias("LONGITUD"),
        "diff_seg",
        "dist_prev_m"
    )

    curr_rows = raros_sample.select(
        "pair_id",
        F.lit("curr").alias("tipo_fila"),
        "id_tarjeta",
        "fecha",
        "timestamp_validacion",
        "cod_em_operadora",
        "cod_perfil",
        "id_tipo_titulo",
        "tipo_transporte",
        "punto_acceso",
        "nombre_parada",
        "LATITUD",
        "LONGITUD",
        "diff_seg",
        "dist_prev_m"
    )

    pairs_raros = (
        prev_rows
        .unionByName(curr_rows)
        .orderBy("pair_id", "tipo_fila")
    )

    
    '''8A) Eliminación de todas las transacciones de una tarjeta cuando algún 
    par consecutivo de transacciones presenta incoherencias en su registroen el día virtual considerado'''

    df_sin_dup = df.filter(~F.col("es_duplicado"))

    bad_groups = (
        df_sin_dup
        .filter(F.col("caso_raro_distinta_estacion") == True)
        .select("id_tarjeta", "fecha")
        .distinct()
    )
    
    df_sin_raros = df_sin_dup.join(bad_groups, on=["id_tarjeta", "fecha"], how="left_anti")    

    
    cols_out = [
        "id_tarjeta",
        "id_resultado",
        'cod_em_operadora',
        "timestamp_validacion",
        "fecha",
        "tipo_transporte",
        "cod_perfil",
        "id_tipo_titulo",
        "nombre_parada",
        "punto_acceso",
        "LATITUD",
        "LONGITUD"
    ]

    return (
        df_sin_raros.select(cols_out),
        df_raros.select(cols_out),
        pairs_raros
    )
    
from pyspark.sql import functions as F

   
    

def eliminacion_validaciones_unicas_en_un_dia(df):
    '''POST DIA VIRTUAL!!!
    
    Esta decisión se debe a la necesidad por parte del método
trip chaining de disponer de más de 1 transacción para poder reconstruir los viajes
realizados con la tarjeta de transporte

        esto se debe hacer TRAS HABER ELIMINADO DUPLICADOS'''

    df.createOrReplaceTempView("mi_tabla")
    
    df_resultado = spark.sql("""
        SELECT *
        FROM (
            SELECT
                t.*,
                COUNT(*) OVER (
                    PARTITION BY id_tarjeta,fecha
                ) AS cnt
            FROM mi_tabla t
        ) x
        WHERE cnt > 1
    """)

    return df_resultado

def eliminacion_validaciones_previas_a_primera_subida(df):
    '''POST DIA VIRTUAL!!
    id_resultado equivalente a subida podria tener mas codigos'''

    df.createOrReplaceTempView("validaciones")
    
    df_resultado = spark.sql("""
        SELECT *
        FROM (
            SELECT
                v.*,
                MIN(CASE WHEN id_resultado IN (1, 2, 3, 4, 6, 11, 12, 63, 64, 65, 66) THEN timestamp_validacion END) 
                    OVER (
                        PARTITION BY id_tarjeta, fecha
                    ) AS primera_subida
            FROM validaciones v
        ) x
        WHERE timestamp_validacion >= primera_subida
          AND primera_subida IS NOT NULL
    """)

    return df_resultado

def eliminacion_validaciones_donde_tarjeta_una_sola_subida_en_dia(df):
    '''POST DIA VIRTUAL: >=2 subidas por (id_tarjeta, dia_virtual).'''
    df.createOrReplaceTempView("validaciones")

    df_unica_subida_en_dia = spark.sql("""
        SELECT *
        FROM (
            SELECT
                v.*,
                SUM(
                    CASE 
                        WHEN id_resultado IN (1, 2, 3, 4, 6, 11, 12, 63, 64, 65, 66)
                        THEN 1 ELSE 0 
                    END
                ) OVER (PARTITION BY id_tarjeta,fecha) AS num_subidas_dia
            FROM validaciones v
        ) x
        WHERE num_subidas_dia == 1
    """)

    df_resultado = spark.sql("""
        SELECT *
        FROM (
            SELECT
                v.*,
                SUM(
                    CASE 
                        WHEN id_resultado IN (1, 2, 3, 4, 6, 11, 12, 63, 64, 65, 66)
                        THEN 1 ELSE 0 
                    END
                ) OVER (PARTITION BY id_tarjeta, fecha) AS num_subidas_dia
            FROM validaciones v
        ) x
        WHERE num_subidas_dia > 1
    """)
    return df_resultado,df_unica_subida_en_dia

from pyspark.sql import functions as F

def eliminar_tarjetas_excepcion_completas_sql(df, umbral=20):

    df.createOrReplaceTempView("v")

    tarjetas_malas = spark.sql(f"""
        SELECT DISTINCT id_tarjeta
        FROM (
            SELECT id_tarjeta, fecha, COUNT(*) AS n
            FROM v
            GROUP BY id_tarjeta, fecha
        ) t
        WHERE n > {umbral}
    """)
    return df.join(tarjetas_malas, on="id_tarjeta", how="left_anti")

def eliminar_validaciones_pacs_l7b_sql(df, pac_col="punto_acceso"):
    """
    Elimina filas cuyo pac_col esté en el conjunto L7B y actualmente tb 02_L7_P55.
    """
    df.createOrReplaceTempView("v")

    pacs_a_borrar = [
        "02_L7_P56",
        "02_L7_P57",
        "02_L7_P58",
        "02_L7_P59",
        "02_L7_P60",
        "02_L7_P55",
        "04_L0_P64",
        '04_L0_P20'
    ]

    values_sql = " UNION ALL ".join([f"SELECT '{p}' AS pac" for p in pacs_a_borrar])

    pacs_borrar = spark.sql(values_sql)
    pacs_borrar.createOrReplaceTempView("pacs_borrar")

    res = spark.sql(f"""
        SELECT v.*
        FROM v
        LEFT ANTI JOIN pacs_borrar p
        ON v.{pac_col} = p.pac
    """)

    return res

def sobrescribir_pac_coslada_sql(df, pac_col="punto_acceso"):
    """
    Asumo que esta gente entra al CERCANIAS. Reemplaza 02_L7_P55 -> 04_L0_P18 en la columna pac_col.
    """
    return df.withColumn(
        pac_col,
        F.when(F.col(pac_col) == F.lit("02_L7_P55"), F.lit("04_L0_P18"))
         .otherwise(F.col(pac_col))
    )

def borrar_codigos_inusuales(df):
    """CUBO IZQ no habla en absoluto de id_resultado, por lo que haremos la suposicion de que no hay validaciones con id_resutado inusual, solo contaremos con validaciones que lo tengan en (1,2,44,4,45,9) -> Tratar eso va a ser una aportación 
    codigos_raros = ['81', '79', '34','73','75','77'] 
    n_antes = df.count()
    df =  df.filter(~F.col('id_resultado').isin(codigos_raros))
    n_despues = df.count()
    n_borradas = n_antes - n_despues
    print(f'se han eliminado {n_borradas} codigos raros\n')"""
    return df

def filtrar_perfil_unico(df,codigo):
    """nos quedamos solo con las validaciones con un cod perfil determinado"""
    
    return df.filter(F.col('cod_perfil') == F.lit(codigo))

    

    

'''Eliminación de todas las transacciones de una tarjeta cuando algún
par consecutivo de transacciones no es reordenable---> lo de mi validacion metro-cercanias famosa'''

def pipeline_tratamiento(path_csv,hiperparams):

    df_base, df_raros, pairs_raros = crear_dfs_y_descartar_duplicados(path_csv,hiperparams)

    

    

    df_sin_tarjetas_excepcionales = eliminar_tarjetas_excepcion_completas_sql(df_base, umbral= 20)

    df_sin_previas = eliminacion_validaciones_previas_a_primera_subida(df_sin_tarjetas_excepcionales)

    if hiperparams['perfil_unico'] in ['01','02','03','0B','04','06']:
        print(f'filtrado solo me quedo con validaciones con cod perfil {hiperparams['perfil_unico']}\n')
        df_sin_previas = filtrar_perfil_unico(df_sin_previas,hiperparams['perfil_unico'])

    df_sin_metro_l7b = eliminar_validaciones_pacs_l7b_sql(df_sin_previas)

    df_final,df_unica_subida_en_dia = eliminacion_validaciones_donde_tarjeta_una_sola_subida_en_dia(df_sin_metro_l7b)

    if hiperparams['paper_original'] == 'munizaga':
        df_final = df_sin_metro_l7b

    df_final = tratamiento_zona_tarifaria(df_final)

    print(f'numero de transacciones tras el procesamiento ANTES DE QUITAR LAS TRANSACCIONES CON PUNTOS DE ACCESO QUE DESCONOCEMOS: {df_final.count()}')

    df_final = df_final.filter(F.col("tipo_transporte").isNotNull())

    print(f'numero de transacciones tras el procesamiento TRAS QUITAR TODO LO MALO: {df_final.count()}')
    

    _PATHS_SPARK_ONLY = {
        str(_BASE_DIR / "prediccion_descenso" / "subidas_enero_2025.csv"),
        str(_BASE_DIR / "historial_enero_2025.csv"),
        str(_BASE_DIR / "historial_febrero_2025.csv"),
    }
    if path_csv not in _PATHS_SPARK_ONLY:

        df_final = df_final.withColumn("primera_subida", F.col("primera_subida").cast("string"))
        df_final = df_final.toPandas()

        df_final["primera_subida"] = pd.to_datetime(df_final["primera_subida"], errors="coerce", utc=True)
        df_final = df_final.sort_values(["id_tarjeta", "timestamp_validacion"]).reset_index(drop=True)

    
    
    return df_final, df_raros, pairs_raros,df_unica_subida_en_dia

def guardar_df_csv(df, nombre_salida="df_procesado"):
    """
    Guarda el DataFrame de Spark como CSV en la ruta actual.
    Crea una carpeta `nombre_salida` con un único part-*.csv dentro.
    """
    ruta_actual = os.getcwd()
    ruta_salida = os.path.join(ruta_actual, nombre_salida)

    (df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(ruta_salida)
    )

