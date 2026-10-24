from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sys
from pyspark import StorageLevel
sys.path.append("/home/jupyter-lquijano/david")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import broadcast

def cargar_mes_anterior(self):
    path = "/home/jupyter-lquijano/david/estudios/cubo_izquierdo/prediccion_descenso/50mil_tarjetas_subidas_enero"

    self.subidas_mes_anterior = (
        self.spark.read
        .parquet(path)
        .select(
            "id_tarjeta",
            "timestamp_validacion",
            "punto_acceso",
            "nombre_parada",
            "LATITUD",
            "LONGITUD",
            "tipo_transporte",
            "cod_em_operadora"
        )
    )

    '''df_peque = df_peque.withColumn(
    "zonas_permitidas",
    F.from_json(F.col("zonas_permitidas"), ArrayType(StringType()))
    )'''

def get_puntos_frecuentados(self, df):
    '''el df incluye las subidas muy especiales unknown y subidas que unicas en el dia que tampoco tenian una subida el dia siguiente
    1.cojo las subidas del mes anteriorr de las tarjetas q hay en df.
    2.crear col franja_temporal
    3.agrupamos por tarjeta y franjas y semana (son solo dos y amplias, segun el paper 4am-12am y 3pm-2am)
    4. crear df con los puntos mas frecuentados por tarjeta en el mes de los datos
    '''
    self.cargar_mes_anterior()
    tarjetas = df.select("id_tarjeta").distinct()
    self.subidas_mes_anterior = self.subidas_mes_anterior.join(
        tarjetas,
        on="id_tarjeta",
        how="left_semi"
    )

    subidas_historicas = (
        self.subidas_mes_anterior
        .withColumn("hora", F.hour("timestamp_validacion"))
        .withColumn(
            "franja",
            F.when((F.col("hora") >= 4) & (F.col("hora") < 12), "1a")
             .when((F.col("hora") >= 15) | (F.col("hora") <= 2), "2a")
             .otherwise("entremedias")
        )
        .drop("hora")
        .withColumn("semana", F.date_trunc("week", F.col("timestamp_validacion")))
        .localCheckpoint(eager=True)
    )

    puntos_recurrentes = (
        subidas_historicas
        .groupBy("semana", "franja", "id_tarjeta", "punto_acceso")
        .agg(F.count("*").alias("n_validaciones"))
        .filter(F.col("n_validaciones") >= 3)
        .dropDuplicates(["id_tarjeta", "punto_acceso", "franja"])
    )

    meta_puntos = (
        subidas_historicas
        .select(
            "id_tarjeta",
            "punto_acceso",
            "franja",
            "nombre_parada",
            "LATITUD",
            "LONGITUD",
            "tipo_transporte",
            "cod_em_operadora"
        )
        .dropDuplicates(["id_tarjeta", "punto_acceso", "franja"])
    )

    self.puntos_frecuentados = (
        puntos_recurrentes
        .join(
            meta_puntos,
            on=["id_tarjeta", "punto_acceso", "franja"],
            how="left"
        )
        .withColumn("recurrencia", F.lit(True))
        .localCheckpoint(eager=True)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    print("FIN GET PTS FRECUENTADOS")
def hay_recurrencia(self, df):
    """cojo las subidas tan raras y veo si les voy a aplicar inferencia(si son en subida frecuentada ) o no"""
    
    df = (
        df
        .withColumn("hora", F.hour("timestamp_validacion"))
        .withColumn(
            "franja",
            F.when((F.col("hora") >= 4) & (F.col("hora") < 12), "1a")
             .when((F.col("hora") >= 15) | (F.col("hora") <= 2), "2a")
             .otherwise("entremedias")
        )
        .drop("hora")
    )
        
    df = (df
         .join(self.puntos_frecuentados.select("id_tarjeta", "punto_acceso", "franja", "recurrencia"),
                on = ["id_tarjeta", "punto_acceso","franja"],how='left')
         .withColumn("recurrencia", F.coalesce(F.col("recurrencia"), F.lit(False)))
         .localCheckpoint(eager=True)
         )

    self.df_no_predecible = (df.filter(F.col('recurrencia') == False).localCheckpoint(eager=True).persist(StorageLevel.MEMORY_AND_DISK))
    self.df_a_inferir = (df.filter(F.col('recurrencia') == True).localCheckpoint(eager=True).persist(StorageLevel.MEMORY_AND_DISK))
    df.unpersist()

def creando_lote_segunda_sub(self):

    a_base = self.df_a_inferir.select(
        "pair_id",
        "id_tarjeta",
        "punto_acceso",
        "franja",
        "timestamp_validacion",
        "fecha",
        "cod_perfil",
        "id_tipo_titulo",
        "id_resultado",
        "zonas_permitidas",
        "num_subidas_dia",
        "num_viaje",
        "n_validaciones_dia",
        "primera_subida",
        "n_por_fecha"
    )

    a_base = a_base.localCheckpoint(eager=True)

    a_entremedias = a_base.filter(~F.col("franja").isin("1a", "2a"))

    a = (
        a_base
        .filter(F.col("franja").isin("1a", "2a"))
        .withColumn("_id_fila_a", F.monotonically_increasing_id())
        .withColumn(
            "franja_objetivo",
            F.when(F.col("franja") == "1a", F.lit("2a"))
             .when(F.col("franja") == "2a", F.lit("1a"))
        )
    )
    

    b = (
        self.puntos_frecuentados
        .select(
            "id_tarjeta",
            "franja",
            "punto_acceso",
            "nombre_parada",
            "LATITUD",
            "LONGITUD",
            "tipo_transporte",
            "cod_em_operadora"
        )
        .dropDuplicates(["id_tarjeta", "franja", "punto_acceso"])
    )
  

    b = broadcast(b)

    joined = (
        a.alias("a")
        .join(
            b.alias("b"),
            (
                (F.col("a.id_tarjeta") == F.col("b.id_tarjeta")) &
                (F.col("a.franja_objetivo") == F.col("b.franja")) &
                (F.col("a.punto_acceso") != F.col("b.punto_acceso"))
            ),
            "left"
        )
    )

    w = Window.partitionBy(F.col("a._id_fila_a")).orderBy(F.lit(1))

    joined_1 = (
        joined
        .withColumn("rn", F.row_number().over(w))
        .localCheckpoint(eager=True)
    )

    print(3)

    punto_recurrente_por_fila = (
        joined_1
        .filter((F.col("rn") == 1) & F.col("b.punto_acceso").isNotNull())
        .select(
            F.col("a.pair_id"),
            F.col("a.id_tarjeta"),
            F.col("b.punto_acceso"),
            F.col("b.nombre_parada"),
            F.col("b.LATITUD"),
            F.col("b.LONGITUD"),
            F.col("b.tipo_transporte"),
            F.col("b.cod_em_operadora"),
            F.col("a.cod_perfil"),
            F.col("a.id_tipo_titulo"),
            F.col("a.id_resultado"),
            F.col("a.zonas_permitidas"),
            (F.col("a.timestamp_validacion") + F.expr("INTERVAL 4 HOURS")).alias("timestamp_validacion"),
            F.col("a.fecha"),
            F.col("a.num_subidas_dia"),
            F.col("a.num_viaje"),
            F.col("a.n_validaciones_dia"),
            F.col("a.primera_subida"),
            F.col("a.n_por_fecha")
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    filas_sin_match = (
        joined_1
        .filter((F.col("rn") == 1) & F.col("b.punto_acceso").isNull())
        .select("a.*")
        .drop("_id_fila_a", "franja_objetivo")
    )
    
    old_no_pred = self.df_no_predecible

    self.df_no_predecible = (
        self.df_no_predecible
        .unionByName(filas_sin_match, allowMissingColumns=True)
        .unionByName(a_entremedias, allowMissingColumns=True)
        .localCheckpoint(eager=True)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    
    if old_no_pred is not None:
        old_no_pred.unpersist()

    self.puntos_frecuentados.unpersist()

    print(5)

    return punto_recurrente_por_fila, self.df_no_predecible
    

def prediccion_descenso_lee(self):
    '''especial solo para predecir non continuous-recurrent travel del paper de Lee 2021
    (subidas sin 2a sub tras buscar con el next day + subidas unknown de primeras en la 1a inferencia)'''

    seg2 = (
    self.lote_segunda_sub_lee.select(
            "pair_id",
            F.col("punto_acceso").alias("punto_acceso_2sub"),
            F.col("tipo_transporte").alias("tipo_transporte_2sub"),
            F.col("zonas_permitidas").alias("zonas_permitidas"),
            F.col("cod_perfil").alias("cod_perfil")
        )
        .localCheckpoint(eager=True).persist(StorageLevel.MEMORY_AND_DISK)
    )

    

    vecinas = (
        seg2.alias("s2")
        .join(
            self.fichero_vecinas.alias("f"),
            F.col("s2.punto_acceso_2sub") == F.col("f.punto_acceso"),
            how="left"
        )
        .select(
            "s2.pair_id",
            "s2.punto_acceso_2sub",
            "s2.tipo_transporte_2sub",
            "s2.zonas_permitidas",
            "s2.cod_perfil",
            F.col("f.vecina").alias("vecina"),
            F.col("f.tipo_transporte_vecina").alias("tipo_transporte_vecina"),
            F.col("f.distancia_m").cast("double").alias("distancia_m"),
            F.col("f.distancia_OSM").cast("double").alias("distancia_OSM")
            
        )
    )

    candidatas = vecinas

    

    
    candidatas= candidatas.filter(F.col("vecina").isNotNull())
    
        

    candidatas = self.quitar_1sub_de_candidatas(candidatas)
    
    candidatas = candidatas.filter(F.col("tipo_transporte_vecina") == F.col("tipo_transporte"))

    if 3 in self.hipotesis:
        candidatas = self.radio_busqueda_por_perfil(candidatas)
        
    

    

    candidatas = candidatas.localCheckpoint(eager=True).persist(StorageLevel.MEMORY_AND_DISK)

    

    cand_metro = candidatas.filter(F.col("tipo_transporte").isin("metro", "cercanias"))

    if (1 in self.hipotesis) or (10 in self.hipotesis):
        cand_metro = self.priorizar_linea_de_subida(cand_metro)
    
    if 7 in self.hipotesis:
        cand_metro = self.filtrar_por_zona_tarifaria(cand_metro)
    

    cand_bus_ml = candidatas.filter(~F.col("tipo_transporte").isin("metro", "cercanias"))
    

    cand_bus_ml = self.filtrar_por_misma_linea(cand_bus_ml)

    cand_bus_ml = self.filtrar_por_sentido(cand_bus_ml)
    

    finalistas = cand_metro.unionByName(cand_bus_ml, allowMissingColumns=True)
    finalistas = finalistas.localCheckpoint(eager=True).persist(StorageLevel.MEMORY_AND_DISK)

    

    base = self.lote_primera_sub_lee.select("pair_id").localCheckpoint(eager=True)
    
    n_finalistas_df = finalistas.groupBy("pair_id").agg(F.count("*").alias("n_finalistas"))

    if 8 in self.hipotesis:
        print('OSM')
        dist_orden = F.col("distancia_OSM")
    elif 9 in self.hipotesis:
        print('OSM + dist normal')
        dist_orden = (
            F.coalesce(F.col("distancia_m"), F.lit(0.0)) +
            F.coalesce(F.col("distancia_OSM"), F.lit(0.0)) * 0.15
        )
    else:
        print('dist normal')
        dist_orden = F.col("distancia_m")

    finalistas_ranked = (
        finalistas
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("pair_id").orderBy(dist_orden.asc_nulls_last())
            )
        )
    )
    finalistas_ranked = finalistas_ranked.localCheckpoint(eager=True)

    ganadora = (
        finalistas_ranked
        .filter(F.col("rn") == 1)
        .select("pair_id", F.col("vecina").alias("prediccion_descenso"), 
                 F.col("distancia_m").alias("ganadora_m"),
                 F.col("distancia_OSM").alias("ganadora_OSM"),
               )
    )
    

    segunda_candidata = (
        finalistas_ranked
        .filter(F.col("rn") == 2)
        .select("pair_id", F.col("vecina").alias("2a_candidata"), 
                 F.col("distancia_m").alias("2a_candidata_m"),
                 F.col("distancia_OSM").alias("2a_candidata_OSM"),
    ))
                                                
                                                
                                                
    
    
    pred_final = (
        base
        .join(n_finalistas_df, "pair_id", "left")
        .fillna({"n_finalistas": 0})
        .join(ganadora, "pair_id", "left")
        .join(segunda_candidata, "pair_id", "left")
        .withColumn("prediccion_descenso", F.coalesce(F.col("prediccion_descenso"), F.lit("unknown")))
        .withColumn("2a_candidata", F.coalesce(F.col("2a_candidata"), F.lit("unknown")))
        .withColumn("ganadora_m", F.coalesce(F.col("ganadora_m"), F.lit("unknown")))
        .withColumn("ganadora_OSM", F.coalesce(F.col("ganadora_OSM"), F.lit("unknown")))
        .withColumn("2a_candidata_m", F.coalesce(F.col("2a_candidata_m"), F.lit("unknown")))
        .withColumn("2a_candidata_OSM", F.coalesce(F.col("2a_candidata_OSM"), F.lit("unknown")))
    )
    pred_final = pred_final.localCheckpoint(eager=True)

    '''pred_final = pred_final.replace('02_L12_P22', '02_L3_P58', subset=['prediccion_descenso', '2a_candidata'])'''
    
   
    predicciones = pred_final.select("pair_id", "prediccion_descenso", '2a_candidata','ganadora_m','ganadora_OSM','2a_candidata_m','2a_candidata_OSM')
    predicciones = predicciones.localCheckpoint(eager=True)
    old_lote = self.lote_primera_sub_lee

    self.lote_primera_sub_lee = (
        self.lote_primera_sub_lee
        .join(predicciones, on="pair_id", how="left")
        .withColumn("tipo_asignacion_2sub", F.lit("lee"))
        .localCheckpoint(eager=True)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    
    if old_lote is not None:
        old_lote.unpersist()
        

    orden = ["id_tarjeta", "fecha", "timestamp_validacion", "punto_acceso"]

    df_unknown = self.lote_primera_sub_lee.filter(F.col('prediccion_descenso') == 'unknown') 
    pd_unknown = df_unknown.orderBy(*orden).toPandas().sort_values(["id_tarjeta", "timestamp_validacion"]).reset_index(drop=True)

    
   

    print("ANTES pd_lote_primera_sub")
    pd_lote_primera_sub = self.lote_primera_sub_lee.orderBy(*orden).toPandas()
    
    print("ANTES pd_lote_segunda_sub")
    pd_lote_segunda_sub = self.lote_segunda_sub_lee.toPandas()
    
    print("ANTES reorder pandas")
    pd_lote_segunda_sub = (
        pd_lote_segunda_sub.set_index("pair_id")
        .loc[pd_lote_primera_sub["pair_id"]]
        .reset_index()
    )
    print('ha pasado!!!')

    assert (pd_lote_primera_sub["pair_id"].values == pd_lote_segunda_sub["pair_id"].values).all()
    
   
    if len(pd_lote_primera_sub) != len(pd_lote_segunda_sub):
        raise ValueError("lote_primera_sub y lote_segunda_sub no tienen la misma longitud")
        
            
    pred_series = pd_lote_primera_sub["prediccion_descenso"].copy()
    '''pd_lote_primera_sub["punto_acceso_2sub"] = pd_lote_segunda_sub["punto_acceso"].to_numpy()
    pd_lote_primera_sub_sin_pred = pd_lote_primera_sub.drop(
        columns=["prediccion_descenso"],
        errors="ignore"   # por si algún día no existe, no rompe
    )'''

  
    
    return pd_lote_primera_sub, pd_lote_segunda_sub, pred_series, pd_unknown

def check_mismas_columnas(self,df_base, df_nuevo, nombre_base="df_base", nombre_nuevo="df_nuevo"):
    cols_base = list(df_base.columns)
    cols_nuevo = list(df_nuevo.columns)

    set_base = set(cols_base)
    set_nuevo = set(cols_nuevo)

    faltan_en_nuevo = [c for c in cols_base if c not in set_nuevo]
    sobran_en_nuevo = [c for c in cols_nuevo if c not in set_base]

    print(f"\n--- Comparando {nombre_nuevo} vs {nombre_base} ---")

    if faltan_en_nuevo:
        print("Faltan en nuevo:", faltan_en_nuevo)
    if sobran_en_nuevo:
        print("Sobran en nuevo:", sobran_en_nuevo)

    if not faltan_en_nuevo and not sobran_en_nuevo:
        print("Tienen las mismas columnas por nombre.")

        if cols_base == cols_nuevo:
            print("Además, están en el mismo orden.")
        else:
            print("Mismas columnas, pero distinto orden.")
            print("Orden base :", cols_base)
            print("Orden nuevo:", cols_nuevo)

    assert set_base == set_nuevo, f"{nombre_nuevo} no tiene las mismas columnas que {nombre_base}"

    

        
        
        

        
        
        
        

        
    
    