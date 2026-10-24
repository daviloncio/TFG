
from pyspark.sql.window import Window
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import col, trim
from pyspark.sql.functions import to_timestamp
from pyspark.sql import functions as F
from zona_tarifaria import *
import json
from tqdm.auto import tqdm
from check_transbordo.check_transbordo import *
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from pyspark.sql.types import ArrayType, StringType
import ipdb
from prediccion_descenso.construir_lotes import construir_lotes, construir_lotes_next_day
from prediccion_descenso.descenso_por_puntos_frecuentados import *
from estudios.cubo_izquierdo.preprocesado_cubo_izq import pipeline_tratamiento
from spark_session import spark

class SparkPrediccionDescenso():
    
    def __init__(
        self,
        lote_inicial, 
        hiperparams,
        hipotesis = {}
    ):

        """
        Versión SPARK.
        hipotesis definirá cómo hacemos ciertos pasos para compararlo con la version original
        lote_inicial: pyspark.sql.DataFrame solo con subidas post tratamiento
        
        """

        
        self.spark = spark
        self.hipotesis = hipotesis
        self.hiperparams = hiperparams
        

        self.fichero_vecinas = (
            spark.read.option("header", True)
            .csv(str(BASE_DIR / ".." / ".." / "topologia" / "paradas_vecinas_1500m_OSM.csv"))
            .withColumn("distancia_m", F.col("distancia_m").cast("double"))
            .withColumn("distancia_OSM", F.col("distancia_OSM").cast("double"))
        )

            
        if 3 in self.hipotesis:
            radio_busqueda = F.lit(self.hiperparams['radio_busqueda_turista'])     
        else:
            radio_busqueda = F.lit(self.hiperparams['radio_busqueda'])

        

        self.fichero_vecinas = self.fichero_vecinas.filter(F.col("distancia_m") <= radio_busqueda)
        

        

        self.rutas_inter = (
            spark.read.option("header", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(BASE_DIR /"RESCATE_csv_bien"/ "rutas_interurbano_urbano.csv"))
            .withColumn("route_id", F.col("route_id").cast("string"))
        )
        self.pacs_inter = (
            spark.read.option("header", True)
            .csv(str(BASE_DIR / "RESCATE_csv_bien"/"pac_interurbanos.csv"))
            .withColumn("DPAYPOINT", F.col("DPAYPOINT").cast("string"))
            .withColumn("IDFLINEAGESTRA", F.col("IDFLINEAGESTRA").cast("string"))
            .withColumn("IDFPARADAGESTRA", F.col("IDFPARADAGESTRA").cast("string"))
        )
        
        self.rutas_emt = (
            spark.read.option("header", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(BASE_DIR /"RESCATE_csv_bien"/ "rutas_EMT.csv"))
            .withColumn("route_id", F.col("route_id").cast("string"))
        )

        
        self.pacs_emt = (
            spark.read.option("header", True)
            .csv(str(BASE_DIR / "RESCATE_csv_bien"/"topologia_solo_EMT.csv"))
            .withColumn("DPAYPOINT", F.col("DPAYPOINT").cast("string"))
            .withColumn("IDFLINEAGESTRA", F.col("IDFLINEAGESTRA").cast("int").cast("string"))
            .withColumn("CODIGOPARADAEMT", F.col("CODIGOPARADAEMT").cast("string"))
        )
        
        self.rutas_ml = (
            spark.read.option("header", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(BASE_DIR /"RESCATE_csv_bien"/ "rutas_ml.csv"))
            .withColumn("route_id", F.col("route_id").cast("string"))
        )
        self.pacs_ml = (
            spark.read.option("header", True)
            .csv(str(BASE_DIR /"RESCATE_csv_bien"/ "pacs_ml.csv"))
            .withColumn("DPAYPOINT", F.col("DPAYPOINT").cast("string"))
            .withColumn("IDFLINEAGESTRA", F.col("IDFLINEAGESTRA").cast("string"))
            .withColumn("IDFESTACION", F.col("IDFESTACION").cast("string"))
        )

        self.pacs_trenes = (
            spark.read.option("header", True).csv(str(BASE_DIR /"RESCATE_csv_bien"/ "topologia_trenes2.csv"))
            .withColumn("punto_acceso", F.col("DPAYPOINT").cast("string"))
            .withColumn("CORONATARIFARIA", F.col("CORONATARIFARIA").cast("string"))

        )

        self.tabla_puntos = (
            spark.read.option("header", True)
            .csv(str(BASE_DIR.parent.parent / "topologia" / "tabla_puntos_acceso_con_coords.csv"))
            .select(
                F.col("punto_acceso"),
                F.col("LATITUD").cast("double"),
                F.col("LONGITUD").cast("double"),
            )
            .cache()
        )
        

        

        from pyspark.sql import DataFrame as SparkDataFrame
        if isinstance(lote_inicial, SparkDataFrame):
            self.lote_inicial = lote_inicial
        else:
            lote_inicial["timestamp_validacion"] = pd.to_datetime(lote_inicial["timestamp_validacion"], errors="coerce")
            self.lote_inicial = spark.createDataFrame(lote_inicial)
        '''self.lote_inicial = (
            self.lote_inicial
            .withColumn("fecha", F.to_date("timestamp_validacion"))
        )'''

        w_fecha = Window.partitionBy("id_tarjeta","fecha")
        self.lote_inicial = self.lote_inicial.withColumn("n_por_fecha", F.count("*").over(w_fecha))

        self.df_unico_dia = self.lote_inicial.filter(F.col("n_por_fecha") == 1).drop("n_por_fecha") 
        

        self.df_impredecible_next_day = None
        
        
        self.lote = self.lote_inicial.filter(F.col("n_por_fecha") > 1).drop("n_por_fecha")

        pacs_inter = self.pacs_inter.select(
            F.lit("bus_interurbano").alias("tipo_transporte"),
            F.col("DPAYPOINT").alias("punto_acceso"),
            F.col("IDFLINEAGESTRA").cast("string").alias("line_id"),
            F.col("IDFPARADAGESTRA").cast("string").alias("stop_id"),
        )
    
        pacs_emt = self.pacs_emt.select(
            F.lit("bus_emt").alias("tipo_transporte"),
            F.col("DPAYPOINT").alias("punto_acceso"),
            F.col("IDFLINEAGESTRA").cast("string").alias("line_id"),
            F.col("CODIGOPARADAEMT").cast("string").alias("stop_id"),
        )
    
        pacs_ml = self.pacs_ml.select(
            F.lit("metro_ligero_tranvia").alias("tipo_transporte"),  
            F.col("DPAYPOINT").alias("punto_acceso"),
            F.col("IDFLINEAGESTRA").cast("string").alias("line_id"),
            F.col("IDFESTACION").cast("string").alias("stop_id"),
        )
    
        self.pacs_all = (
            pacs_inter
            .unionByName(pacs_emt)
            .unionByName(pacs_ml)
            .dropDuplicates(["tipo_transporte", "punto_acceso"])
            .cache()
        )

        self.build_route_stops_all()

    
    def quitar_1sub_de_candidatas(self,candidatas):
        """hacemos join con 1sub y se diferencia entre caso especial de tranvia y el resto"""

        
        sub1 = self.lote_primera_sub.select(
            "pair_id",
            F.col("punto_acceso").alias("p_acc_sub1"),
            'tipo_transporte'
        )

        candidatas = candidatas.join(sub1, on="pair_id", how="left")

        es_b1 = F.col("p_acc_sub1").startswith("B1")

        sufijo = F.element_at(F.split(F.col("p_acc_sub1"), "_"), -1)
        l1 = F.concat(F.lit("B1_L1_"), sufijo)
        l2 = F.concat(F.lit("B1_L2_"), sufijo)
        
        return candidatas.filter(
            F.when(
                es_b1,
                (F.col("vecina") != l1) & (F.col("vecina") != l2)
            ).otherwise(
                F.col("vecina") != F.col("p_acc_sub1")
            )
        )

        

        

    def filtrar_por_misma_linea(self, candidatas_rows):
        """
        esot es para buses y metro ligero
          
        Devuelve solo candidatas donde:
          - origen y candidata existen en pacs_all
          - line_id origen == line_id candidata
          - stop_id origen != stop_id candidata
        """
    
        po = broadcast(self.pacs_all).alias("po")
        pc = broadcast(self.pacs_all).alias("pc")
    
        c = candidatas_rows.alias("c")
    
        joined = (
            c

            .join(
                po,
                (F.col("c.tipo_transporte") == F.col("po.tipo_transporte")) &
                (F.col("c.p_acc_sub1") == F.col("po.punto_acceso")),
                "left"
            )

            .join(
                pc,
                (F.col("c.tipo_transporte") == F.col("pc.tipo_transporte")) &
                (F.col("c.vecina") == F.col("pc.punto_acceso")),
                "left"
            )
            .select(
                F.col("c.*"),
                F.col("po.line_id").alias("line_o"),
                F.col("po.stop_id").alias("stop_o"),
                F.col("pc.line_id").alias("line_c"),
                F.col("pc.stop_id").alias("stop_c"),
            )
        )
    

        out = joined.filter(
            F.col("line_o").isNotNull() &
            F.col("line_c").isNotNull() &
            (F.col("line_o") == F.col("line_c")) &
            (F.col("stop_o") != F.col("stop_c"))
        )
    
        return out

    def build_route_stops_all(self):
        arr_schema = ArrayType(StringType())
    
        def to_route_stops(df_rutas, tipo_transporte):

            base = (
                df_rutas
                .withColumn("tipo_transporte", F.lit(tipo_transporte))
                .withColumn("line_id", F.col("route_id").cast("string"))
                .withColumn("variant_key", F.sha2(F.col("ruta_paradas"), 256))

                .withColumn("ruta_arr", F.from_json(F.col("ruta_paradas"), arr_schema))
            )
    
            w = Window.partitionBy("tipo_transporte", "line_id").orderBy("variant_key")
            base = base.withColumn("variant_id", F.dense_rank().over(w))
    

            rs = base.select(
                "tipo_transporte",
                "line_id",
                "variant_id",
                F.posexplode("ruta_arr").alias("pos", "stop_id")
            )
            
            return rs
    
        rs_emt   = to_route_stops(self.rutas_emt,   "bus_emt")
        rs_inter = to_route_stops(self.rutas_inter, "bus_interurbano")
        rs_ml    = to_route_stops(self.rutas_ml,    "metro_ligero_tranvia")

        self.route_stops_all = (
            rs_emt.unionByName(rs_inter).unionByName(rs_ml)
            .cache()
        )

        

    def filtrar_por_sentido_metro_cercanias(self, cand_metro):
        """
        Hip 6: elimina candidatas de metro y cercanías que están en el sentido
        contrario al de la 2ª subida.

        Razonamiento: el viajero sube en A (1ª subida) y vuelve a subir en B
        (2ª subida). El trayecto va de A hacia B. Una candidata C está en el
        sentido correcto si el producto escalar (B−A)·(C−A) ≥ 0, es decir,
        si C se encuentra en el mismo semiplano que B respecto a A.
        Si el producto escalar es negativo, C está en la dirección opuesta a
        donde el viajero reaparece → se descarta.

        Caso degenerado: si dist(A,B) < ~300 m (A y B son casi la misma parada,
        p. ej. viaje de ida y vuelta muy corto), el vector dirección no es fiable
        y NO se aplica el filtro para no perder candidatas válidas.

        Las coordenadas se obtienen de tabla_puntos_acceso_con_coords.csv,
        que cubre todos los modos (36.894 paradas).
        """
        tp = self.tabla_puntos

        coords_A = tp.select(
            F.col("punto_acceso").alias("p_acc_sub1"),
            F.col("LATITUD").alias("lat_A"),
            F.col("LONGITUD").alias("lon_A"),
        )
        coords_B = tp.select(
            F.col("punto_acceso").alias("punto_acceso_2sub"),
            F.col("LATITUD").alias("lat_B"),
            F.col("LONGITUD").alias("lon_B"),
        )
        coords_C = tp.select(
            F.col("punto_acceso").alias("vecina"),
            F.col("LATITUD").alias("lat_C"),
            F.col("LONGITUD").alias("lon_C"),
        )

        result = (
            cand_metro
            .join(broadcast(coords_A), on="p_acc_sub1",       how="left")
            .join(broadcast(coords_B), on="punto_acceso_2sub", how="left")
            .join(broadcast(coords_C), on="vecina",            how="left")
            .withColumn("dAB_lat", F.col("lat_B") - F.col("lat_A"))
            .withColumn("dAB_lon", F.col("lon_B") - F.col("lon_A"))
            .withColumn("dAC_lat", F.col("lat_C") - F.col("lat_A"))
            .withColumn("dAC_lon", F.col("lon_C") - F.col("lon_A"))
            .withColumn("dot",    F.col("dAB_lat") * F.col("dAC_lat")
                                + F.col("dAB_lon") * F.col("dAC_lon"))
            .withColumn("dist_AB_sq", F.col("dAB_lat")**2 + F.col("dAB_lon")**2)
        )

        UMBRAL_SQ = 7.3e-6

        filtered = result.filter(
            F.col("dist_AB_sq").isNull()   |
            F.col("lat_C").isNull()        |
            (F.col("dist_AB_sq") < F.lit(UMBRAL_SQ))  |
            (F.col("dot") >= F.lit(0.0))
        )

        return filtered.drop(
            "lat_A","lon_A","lat_B","lon_B","lat_C","lon_C",
            "dAB_lat","dAB_lon","dAC_lat","dAC_lon","dot","dist_AB_sq"
        )

    def filtrar_por_sentido(self, cand_bus_ml):
        rs = self.route_stops_all
    

        o_pos = rs.select(
            F.col("tipo_transporte").alias("tipo_o"),
            F.col("line_id").alias("line_o2"),
            F.col("variant_id").alias("variant_o"),
            F.col("stop_id").alias("stop_o"),
            F.col("pos").alias("pos_o"),
        ).alias("o")
    

        c_pos = rs.select(
            F.col("tipo_transporte").alias("tipo_c"),
            F.col("line_id").alias("line_c2"),
            F.col("variant_id").alias("variant_c"),
            F.col("stop_id").alias("stop_c"),
            F.col("pos").alias("pos_c"),
        ).alias("c")
    
        cand = cand_bus_ml.withColumn("line_id_tmp", F.col("line_o")).alias("cand")
    
        joined = (
            cand
            .join(
                o_pos,
                (F.col("cand.tipo_transporte") == F.col("o.tipo_o")) &
                (F.col("cand.line_id_tmp") == F.col("o.line_o2")) &
                (F.col("cand.stop_o") == F.col("o.stop_o")),
                "inner"
            )
            .join(
                c_pos,
                (F.col("cand.tipo_transporte") == F.col("c.tipo_c")) &
                (F.col("cand.line_id_tmp") == F.col("c.line_c2")) &
                (F.col("o.variant_o") == F.col("c.variant_c")) &
                (F.col("cand.stop_c") == F.col("c.stop_c")),
                "inner"
            )
            .filter(F.col("o.pos_o") <= F.col("c.pos_c"))
            .select("cand.*")
            .drop("line_id_tmp")
            .dropDuplicates()
        )
    
        return joined

    def filtrar_por_zona_tarifaria(self,cand_metro):
        '''hipotesis 7 '''
        pacs_zona = self.pacs_trenes.select(
            F.col('punto_acceso').alias('vecina'),
            F.col('CORONATARIFARIA'),
        )

        cand_metro = cand_metro.join(pacs_zona, on="vecina", how="left")

        return cand_metro.filter(F.array_contains(F.col("zonas_permitidas"), F.col("CORONATARIFARIA")))

    def radio_busqueda_por_perfil(self,candidatas):

        

        radio_busqueda = (
                F.when(F.col("cod_perfil") == "01", F.lit(self.hiperparams['radio_busqueda_adulto']))
                 .when(F.col("cod_perfil") == "02", F.lit(self.hiperparams['radio_busqueda_3a_edad']))
                 .when(F.col("cod_perfil") == "03", F.lit(self.hiperparams['radio_busqueda_joven']))
                 .when(F.col("cod_perfil") == "04", F.lit(self.hiperparams['radio_busqueda_infantil']))
                 .when(F.col("cod_perfil") == "06", F.lit(self.hiperparams['radio_busqueda_tarjeta_azul']))
                 .when(F.col("cod_perfil") == "0B", F.lit(self.hiperparams['radio_busqueda_turista']))
                 .otherwise(F.lit(self.hiperparams['radio_busqueda_adulto']))
            )

        candidatas = candidatas.filter(F.col("distancia_m") <= radio_busqueda)

        return candidatas

    def priorizar_linea_de_subida(self,cand_metro):
        '''hipotesis 1: en metro si tenemos varias candidatas finales de distintas lineas
        nos quedamos solo con aquellas pertenecientes a la linea de la 1 sub justo antes de decidir la ganadora
        OJO: SI NO NINGUNA LINEA DE LAS CANDIDATAS COINCIDE CON LINEA 1_SUB ENTONCES NO HACEMOS NADA???'''
        pass

        lineas = (
            spark.read.option("header", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(BASE_DIR / "correspondencias_lineas.csv"))
            
        )
        
        lineas =( 
            lineas
            .withColumn("lineas_clean", F.regexp_replace(F.col("lineas"), '""', '"')) \
            .withColumn("lineas_arr", F.from_json(F.col("lineas_clean"), ArrayType(StringType()) ))
                      )

        lin_1sub = lineas.select(
            F.col("punto_acceso").alias("p_acc_sub1"),
            F.col("lineas_arr").alias("lineas_1sub"),
        )
        
        lin_vecina = lineas.select(
            F.col("punto_acceso").alias("vecina"),
            F.col("lineas_arr").alias("lineas_vecina"),
        )

        cand_metro = (
            cand_metro
            .join(lin_1sub, on="p_acc_sub1", how="left")
            .join(lin_vecina, on="vecina", how="left")
        )

        

        empty_arr = F.expr("array()")

        cand_metro = (
            cand_metro

            .withColumn(
                "coincide",
                F.size(
                    F.array_intersect(
                        F.coalesce(F.col("lineas_1sub"), empty_arr),
                        F.coalesce(F.col("lineas_vecina"), empty_arr),
                    )
                ) > 0
            )
        )
        
        w = Window.partitionBy("pair_id")

        if 1 in self.hipotesis:
            
            cand_metro = (
                cand_metro

                .withColumn("hay_coinc", F.max(F.col("coincide").cast("int")).over(w))

                .filter(
                    (~F.col("tipo_transporte").isin("metro", "cercanias")) |
                    (F.col("hay_coinc") == 0) |
                    (F.col("coincide") == True)
                )
                .drop("coincide", "hay_coinc")
            )
        elif 10 in self.hipotesis:

            w_pair = Window.partitionBy("pair_id")
            w_rank = Window.partitionBy("pair_id").orderBy(F.col("distancia_m").asc_nulls_last())
            cand_metro = (
                cand_metro
                .withColumn("rn", F.row_number().over(w_rank))
                .withColumn(
                    "dist_1",
                    F.max(F.when(F.col("rn") == 1, F.col("distancia_m"))).over(w_pair)
                )
                .withColumn(
                    "dist_2",
                    F.max(F.when(F.col("rn") == 2, F.col("distancia_m"))).over(w_pair)
                )
                .withColumn(
                    "caso_ajustado",
                    F.when(
                        F.col("dist_2").isNotNull() & F.col("dist_1").isNotNull(),
                        (F.col("dist_2") - F.col("dist_1")) < F.lit(self.hiperparams['condicion_caso_ajustado'])
                    ).otherwise(F.lit(False))
                )
                .withColumn("hay_coinc", F.max(F.col("coincide").cast("int")).over(w))

                .filter(
                    (F.col("tipo_transporte") != "metro") & (F.col("tipo_transporte") != "cercanias") |
                    (F.col("hay_coinc") == 0) |
                    (F.col("coincide") == True) | 
                    (F.col("caso_ajustado") == False)
                )
                
            )

            
        cand_metro = cand_metro.drop("coincide", "hay_coinc",'caso_ajustado')
        
        return cand_metro
        

        
    

    def spark_predicciones(self):
        '''ya no es un bucle como antes,'''
    

        seg2 = self.lote_segunda_sub.select(
            "pair_id",
            F.col("punto_acceso").alias("punto_acceso_2sub"),
            F.col("tipo_transporte").alias("tipo_transporte_2sub"),
            F.col("zonas_permitidas").alias("zonas_permitidas"),
            F.col("cod_perfil").alias("cod_perfil")
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

    
    
        '''fila_self = seg2.select(
            "pair_id",
            "punto_acceso_2sub",
            "tipo_transporte_2sub",
            "zonas_permitidas",
            "cod_perfil",
            F.col("punto_acceso_2sub").alias("vecina"), #aqui estoy metiendo la misma 2sub
            F.col("tipo_transporte_2sub").alias("tipo_transporte_vecina"),
            F.lit(0.0).cast("double").alias("distancia_m"),
            F.lit(0.0).cast("double").alias("distancia_OSM")
        
    
        candidatas = vecinas.unionByName(fila_self))'''
    
        candidatas = vecinas
    
        
    

        
        candidatas= candidatas.filter(F.col("vecina").isNotNull())
        
            

        candidatas = self.quitar_1sub_de_candidatas(candidatas)
        
        candidatas = candidatas.filter(F.col("tipo_transporte_vecina") == F.col("tipo_transporte"))

        if 3 in self.hipotesis:
            candidatas = self.radio_busqueda_por_perfil(candidatas)

        candidatas = candidatas.localCheckpoint(eager=True)

        
    

        

        cand_metro = candidatas.filter(F.col("tipo_transporte").isin("metro", "cercanias"))

        if (1 in self.hipotesis) or (10 in self.hipotesis):
            cand_metro = self.priorizar_linea_de_subida(cand_metro)

        if 7 in self.hipotesis:
            cand_metro = self.filtrar_por_zona_tarifaria(cand_metro)

        if 6 in self.hipotesis:
            cand_metro = self.filtrar_por_sentido_metro_cercanias(cand_metro)

        '''cand_metro_counts = cand_metro.groupBy("pair_id").count()
        
        base_metro = (
            self.lote_primera_sub
            .filter(F.col("tipo_transporte").isin("metro", "cercanias"))
            .select("pair_id", "tipo_transporte", "punto_acceso")
        )
        
        sin_candidatas = (
            base_metro
            .join(cand_metro_counts, on="pair_id", how="left")
            .fillna({"count": 0})
            .filter(F.col("count") == 0)
        )
        
        sin_candidatas.select("tipo_transporte", "punto_acceso").show(0, False)
        print("metro/cercanias sin candidatas:", sin_candidatas.count())'''
        

        cand_bus_ml = candidatas.filter(~F.col("tipo_transporte").isin("metro", "cercanias"))
        

        cand_bus_ml = self.filtrar_por_misma_linea(cand_bus_ml)
    

        cand_bus_ml = self.filtrar_por_sentido(cand_bus_ml)
        

        finalistas = cand_metro.unionByName(cand_bus_ml, allowMissingColumns=True)

        finalistas = finalistas.localCheckpoint(eager=True)

    
        
    
        base = self.lote_primera_sub.select("pair_id")
        
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

        '''ganadora = ( #ESTO DE GANADORA DEBE DE CAMBIAR SEGUN LA HIPOTESIS!! 
            finalistas
            .withColumn("rn", F.row_number().over(Window.partitionBy("pair_id").orderBy(dist_orden.asc_nulls_last())))
            .filter(F.col("rn") == 1)
            .select("pair_id", F.col("vecina").alias("prediccion_descenso"))
        )
        
        pred_final = (
            base
            .join(n_finalistas_df, "pair_id", "left")
            .fillna({"n_finalistas": 0})
            .join(ganadora, "pair_id", "left")
            .withColumn("prediccion_descenso", F.coalesce(F.col("prediccion_descenso"), F.lit("unknown")))
        )

        pred_final = pred_final.replace('02_L12_P22', '02_L3_P58', subset=['prediccion_descenso'])   ''' 

        '''if self.hiperparams['paper_original'] == 'munizag': #descenso del acc con esto...

            finalistas = (
                finalistas
                .withColumn(
                    "razon_munizaga",
                    F.when(
                        F.col("distancia_m").isNotNull() & (F.col("distancia_m") > 0.5),
                        F.col("distancia_OSM") / F.col("distancia_m")
                    ).otherwise(F.lit(1.0))
                )
                .filter(F.col("razon_munizaga").between(0.98, 2))
            )'''

        finalistas_ranked = (
            finalistas
            .withColumn(
                "rn",
                F.row_number().over(
                    Window.partitionBy("pair_id").orderBy(dist_orden.asc_nulls_last())
                )
            )
        )
        

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

        '''pred_final = pred_final.replace('02_L12_P22', '02_L3_P58', subset=['prediccion_descenso', '2a_candidata'])'''
        
        
        predicciones = pred_final.select("pair_id", "prediccion_descenso", '2a_candidata','ganadora_m','ganadora_OSM','2a_candidata_m','2a_candidata_OSM')
    
        self.lote_primera_sub = (
            self.lote_primera_sub
            .join(predicciones, on="pair_id", how="left")
        )

        '''ipdb.set_trace()
        self.lote_primera_sub = ( #suposicion bajada metropolitano SUBE ACCURACY PERO ES ENGAÑOSO!!!!!
            self.lote_primera_sub
            .withColumn(
                "prediccion_descenso",
                F.when(F.col("punto_acceso") == "02_L7_P54", F.lit("02_L7_P51"))
                 .otherwise(F.col("prediccion_descenso"))
            )
        )'''

        '''#emparejamos por pair id
        seg2_info = self.lote_segunda_sub.select(
        "pair_id",
        F.col("punto_acceso").alias("punto_acceso_2sub"),
        F.col("LATITUD").alias("LATITUD_2sub"),
        F.col("LONGITUD").alias("LONGITUD_2sub"),
        F.col("zonas_permitidas").alias("zonas_permitidas"),
        F.col("cod_perfil").alias("cod_perfil"),
        F.col("timestamp_validacion").alias("ts_2sub"),
        F.col("tipo_asignacion_2sub").alias("tipo_asignacion_2sub")
        )
    
        out_spark = (
            self.lote_primera_sub
            .withColumnRenamed("punto_acceso", "p_acc_sub1")   # si quieres conservarlo con ese nombre
            .join(seg2_info, on="pair_id", how="left")
        )
        
        pd_out = out_spark.toPandas()'''
            
    

        orden = ["id_tarjeta", "fecha", "timestamp_validacion", "punto_acceso"]
    
        df_unknown = self.lote_primera_sub.filter(F.col('prediccion_descenso') == 'unknown') 
        pd_unknown = df_unknown.orderBy(*orden).toPandas().sort_values(["id_tarjeta", "timestamp_validacion"]).reset_index(drop=True)

        if self.hiperparams['paper_original'] == 'munizaga':
            print('munizaga, num de subidas solitas en el dia q no tenian subida el siguiente dia', self.df_impredecible_next_day.count())
        else:
            print('num de subidas solitas en el dia en PRED DESCENSO.py', self.df_unico_dia.count())
            
        print('num de validaciones con pred = unknown', df_unknown.count())
    
    
    

        
       

        pd_lote_primera_sub = self.lote_primera_sub.orderBy(*orden).toPandas()
        
        pd_lote_segunda_sub = self.lote_segunda_sub.toPandas()
        pd_lote_segunda_sub = pd_lote_segunda_sub.set_index("pair_id").loc[pd_lote_primera_sub["pair_id"]].reset_index()
        

        assert (pd_lote_primera_sub["pair_id"].values == pd_lote_segunda_sub["pair_id"].values).all()
        
       
        if len(pd_lote_primera_sub) != len(pd_lote_segunda_sub):
            raise ValueError("lote_primera_sub y lote_segunda_sub no tienen la misma longitud")
            
                
        pred_series = pd_lote_primera_sub["prediccion_descenso"].copy()
        pd_lote_primera_sub['tipo_asignacion_2sub']= pd_lote_segunda_sub['tipo_asignacion_2sub']
        
        

        if self.hiperparams['paper_original'] == 'lee': 

            tmp1 = "/home/jupyter-lquijano/david/estudios/cubo_izquierdo/prediccion_descenso/tmp_lote_primera_sub"
            tmp2 = "/home/jupyter-lquijano/david/estudios/cubo_izquierdo/prediccion_descenso/tmp_lote_segunda_sub"
            tmp3 = "/home/jupyter-lquijano/david/estudios/cubo_izquierdo/prediccion_descenso/tmp_df_unknown"
            
            self.lote_primera_sub.write.mode("overwrite").parquet(tmp1)
            self.lote_segunda_sub.write.mode("overwrite").parquet(tmp2)
            df_unknown.write.mode("overwrite").parquet(tmp3)
            
            self.lote_primera_sub = self.spark.read.parquet(tmp1)
            self.lote_segunda_sub = self.spark.read.parquet(tmp2)
            df_unknown = self.spark.read.parquet(tmp3)
          

            df_unknown = df_unknown.drop( 'prediccion_descenso', '2a_candidata', 'ganadora_m', 'ganadora_OSM', '2a_candidata_m', '2a_candidata_OSM','tipo_asignacion_2sub')
            
            df_lee = (
                df_unknown
                .unionByName(self.df_impredecible_next_day)
                .localCheckpoint(eager=True)
            )
            self.num_df_lee = df_lee.count()
            self.get_puntos_frecuentados(df_lee)
            self.hay_recurrencia(df_lee)

            self.lote_segunda_sub_lee,self.df_no_predecible = self.creando_lote_segunda_sub() 

            ids2 = self.lote_segunda_sub_lee.select("pair_id").distinct()
            self.lote_primera_sub_lee = df_lee.join(ids2, "pair_id", "inner")

            self.lote_segunda_sub_lee = self.lote_segunda_sub_lee.localCheckpoint(eager=True)
            self.lote_primera_sub_lee = self.lote_primera_sub_lee.localCheckpoint(eager=True)

            pd_lote_primera_sub_lee, pd_lote_segunda_sub_lee, pred_series_lee,pd_unknown_lee = self.prediccion_descenso_lee() 
            
            pd_lote_segunda_sub_lee['tipo_asignacion_2sub']= pd_lote_primera_sub_lee['tipo_asignacion_2sub']

            

            self.num_no_predecibles_lee = self.df_no_predecible.count() 
            self.num_unknown_lee = pd_unknown_lee.shape[0]
            
            self.check_mismas_columnas(
                pd_lote_primera_sub,
                pd_lote_primera_sub_lee,
                "pd_lote_primera_sub",
                "pd_lote_primera_sub_lee"
            )
            
            self.check_mismas_columnas(
                pd_lote_segunda_sub,
                pd_lote_segunda_sub_lee,
                "pd_lote_segunda_sub",
                "pd_lote_segunda_sub_lee"
            )
         

            mask_validos = pd_lote_primera_sub["prediccion_descenso"] != "unknown"

            pd_lote_primera_sub = pd_lote_primera_sub.loc[mask_validos].reset_index(drop=True)
            pd_lote_segunda_sub = pd_lote_segunda_sub.loc[mask_validos].reset_index(drop=True)
            pred_series = pred_series.loc[mask_validos].reset_index(drop=True)

            

            pd_lote_primera_sub_lee = pd_lote_primera_sub_lee[pd_lote_primera_sub.columns]

            pd_lote_primera_sub = pd.concat(
                [pd_lote_primera_sub, pd_lote_primera_sub_lee],
                ignore_index=True
            )

            pd_lote_segunda_sub_lee = pd_lote_segunda_sub_lee[pd_lote_segunda_sub.columns]

            pd_lote_segunda_sub = pd.concat(
                [pd_lote_segunda_sub, pd_lote_segunda_sub_lee],
                ignore_index=True
            )
            
            pred_series = pd.concat([pred_series, pred_series_lee], ignore_index=True)

            pd_unknown = pd_unknown_lee

            

        
        pd_lote_primera_sub["punto_acceso_2sub"] = pd_lote_segunda_sub["punto_acceso"].to_numpy()
        

        orden = ["id_tarjeta", "fecha", "timestamp_validacion", "punto_acceso"]
        

        idx_orden = pd_lote_primera_sub.sort_values(orden, kind="mergesort").index
        

        pd_lote_primera_sub = pd_lote_primera_sub.loc[idx_orden].reset_index(drop=True)
        pd_lote_segunda_sub = pd_lote_segunda_sub.loc[idx_orden].reset_index(drop=True)
        pred_series = pred_series.loc[idx_orden].reset_index(drop=True)

        pd_lote_primera_sub_sin_pred = pd_lote_primera_sub.drop(
            columns=["prediccion_descenso"],
            errors="ignore"
        )

        
        return pd_lote_primera_sub_sin_pred, pd_lote_segunda_sub, pred_series, pd_unknown

        

SparkPrediccionDescenso.construir_lotes = construir_lotes
SparkPrediccionDescenso.construir_lotes_next_day = construir_lotes_next_day
SparkPrediccionDescenso.creando_lote_segunda_sub = creando_lote_segunda_sub
SparkPrediccionDescenso.get_puntos_frecuentados = get_puntos_frecuentados
SparkPrediccionDescenso.hay_recurrencia = hay_recurrencia
SparkPrediccionDescenso.check_mismas_columnas = check_mismas_columnas
SparkPrediccionDescenso.prediccion_descenso_lee = prediccion_descenso_lee
SparkPrediccionDescenso.cargar_mes_anterior = cargar_mes_anterior

    

    
