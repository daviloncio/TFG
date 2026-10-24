import os
import subprocess

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

print(subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode())
from check_transbordo import *
from crear_mapeos import *
from spark_crear_mapeos import spark_crear_mapeos
from prediccion_descenso.spark_PrediccionDescenso import *
from check_transbordo.check_transbordo import *
from check_transbordo.spark_check_transbordo import SparkCheckTransbordo
from check_transbordo.transbordos_implicitos import detectar_transbordos_implicitos, asignar_transbordo
from crear_matrices import crear_matriz_od
from crear_matrices_transbordo import (
    crear_csv_rutas_transbordo,
    crear_matriz_transbordo_macro,
    crear_matriz_transbordo_micro,
)
from accuracy_bajadas import *
from reconstruir_viajes import *
from preprocesado_cubo_izq import *
from show_side_by_side import *
import time

from pathlib import Path
from prediccion_descenso.PrediccionDescenso import *
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
from prediccion_descenso.construir_lotes import *

spark._jvm.org.apache.log4j.LogManager \
    .getLogger("org.apache.spark.sql.catalyst.util.StringUtils") \
    .setLevel(spark._jvm.org.apache.log4j.Level.ERROR)

class Pipeline():
    """esta clase se sirve de las funciones de preporcesado, inferencia de bajada,
    comprobacion de transbordo y """
    def __init__(self,path = './crudo_validaciones_david.csv',
                 hipotesis = {},
                 is_csv: bool = True,
                 mapeos:bool = False,
                 write_results:bool = False,
                 hiperparams =
                {'radio_busqueda': 650,
                 'MTT': 18*60,
                 'peso_tiempos_metros': 1.1,
                 'peso_tiempo_cercanias': 1.2,
                 'radio_busqueda_joven': 900,
                 'radio_busqueda_adulto': 750,
                 'radio_busqueda_3a_edad': 600,

                 'tipo_validacion': 'BAJADA'
                },
                 name_out= None,
                 mdlf_params: dict = {},
                 ):
                 

        
        self.hiperparams = hiperparams
        self.hipotesis = hipotesis
        self.validaciones = None
        self.mapeos = None
        self.viajes = None
        self.unlinked = None
        self.resultados = None
        self.is_csv = is_csv
        self.name_out = name_out
        self.path = path
        self.almacen = {}
        self.x_pred_checked = None
        self.transbordos_implicitos = None
        self.x_pred = None
        self.x_1sub = None
        self.x_2sub = None
        self.y_pred = None
        self.y_test = None
        self.x_pred_2sub = None
        self.subidas = None
        self.unknown = None
        self.write_results = write_results
        self.mdlf_params = mdlf_params

        
    def load(self): 
        """no spark"""
        
        
        if self.is_csv:
            self.validaciones = pd.read_csv(self.path)

        else: 
            pass

    def _fmt_secs(self,seconds: float) -> str:
        s = int(round(seconds))
        mins = s // 60
        secs = s % 60
        return f"Completado después de {mins} mins e {secs} secs"

        
    def start(self):

        """tratamiento si es necesario, creamos el conjunto de validación, sacamos predicciones, comprobamos_alcanzabilidad"""
    
        t_total0 = time.time()
    

        t0 = time.time()
        print(f'Empezando el preprocesado de {self.path}...')
        self.validaciones, raro, _, self.subidas_unicas_en_dia = pipeline_tratamiento(self.path,self.hiperparams)
        codigos_descenso = ('9', '44', '45','81', '79', '34','73','75','77','10')
        self.subidas = self.validaciones.loc[~self.validaciones['id_resultado'].isin(codigos_descenso)].copy()

        self.subidas = self.subidas.drop_duplicates(
            subset=["id_tarjeta", "timestamp_validacion"], keep="first"
        ).reset_index(drop=True)
        print(self._fmt_secs(time.time() - t0))
        print('raroooo: ',raro)
        
    

        t0 = time.time()
        print(f'Construyendo los mapeos de subida-bajada...')
        if 5 in self.hipotesis:
            self.x_test, self.y_test = spark_crear_mapeos(self.validaciones, self.hiperparams)
        else:
            self.x_test, self.y_test = crear_mapeos(self.validaciones, self.hiperparams)
        print(self._fmt_secs(time.time() - t0))
    
        
        print(
            f'tamaño conjunto validación que se verá reducido pq no todos los pares cumplen '
            f'loas condiciones para aplicar trip chaning:{self.y_test.shape[0]}'
        )
        
    

        t0 = time.time()
        print(f'Inferencia de parada de descenso...')

        if 5 in self.hipotesis:
            instancia = SparkPrediccionDescenso(self.subidas, hipotesis=self.hipotesis,hiperparams = self.hiperparams)
            instancia.lote_inicial = instancia.lote_inicial.localCheckpoint(eager=True)
            
            if self.hiperparams['paper_original'] in ['munizaga','lee']:
                self.subidas_unicas_en_dia_sin_pred = instancia.construir_lotes_next_day()
            else: 
                instancia.construir_lotes()

            instancia.lote_primera_sub = instancia.lote_primera_sub.localCheckpoint(eager=True)
            instancia.lote_segunda_sub = instancia.lote_segunda_sub.localCheckpoint(eager=True)
            
            self.x_1sub, self.x_2sub, self.y_pred, self.unknown = instancia.spark_predicciones()
            if self.hiperparams['paper_original'] == 'lee':
                self.df_no_predecible_next_day  = instancia.df_no_predecible
                
        else: 
            instancia = PrediccionDescenso(self.subidas, hipotesis=self.hipotesis)
            if self.hiperparams['paper_original'] not in [ 'munizaga','lee']:
                instancia.construir_lotes()
            else: 
                self.subidas_unicas_en_dia_sin_pred = instancia.construir_lotes_next_day()
                
            self.x_1sub, self.x_2sub, self.y_pred, self.unknown = instancia.bucle_predecciones()
       
        print(F"TIEMPO TOTAL SECCION PREDICCION: {self._fmt_secs(time.time() - t0)}\n")
        
    

        self.x_1sub["prediccion_descenso"] = self.y_pred.to_numpy()
        t0 = time.time()
        print(f'Comprabando qué es transbordo, bajada y unlinked...')
        if 5 in self.hipotesis:
            checker = SparkCheckTransbordo()
            self.x_pred_checked = checker.is_transbordo(self.x_1sub, self.x_2sub)
        else:
            checker = CheckTransbordo(hipotesis=self.hipotesis)
            self.x_pred_checked = checker.is_transbordo(self.x_1sub, self.x_2sub, self.y_pred)
        print(self._fmt_secs(time.time() - t0))
        self.x_pred_checked.head(5)
            
        

        if 12 in self.hipotesis:
            t0 = time.time()
            print('Detectando y asignando transbordos implícitos...')
            transbordos = detectar_transbordos_implicitos(self.x_pred_checked)
            _df_ti = asignar_transbordo(transbordos)
            _ts_cols = [f for f, t in _df_ti.dtypes if t == "timestamp"]
            for _c in _ts_cols:
                _df_ti = _df_ti.withColumn(_c, F.col(_c).cast("string"))
            self.transbordos_implicitos = _df_ti.toPandas()
            for _c in _ts_cols:
                self.transbordos_implicitos[_c] = pd.to_datetime(self.transbordos_implicitos[_c])
            print(self._fmt_secs(time.time() - t0))

            t0 = time.time()
            print('Creando matrices OD...')
            matrices_dir = Path(__file__).resolve().parent / "matrices_OD"
            out_subdir = matrices_dir / Path(self.path).stem

            crear_matriz_od(self.x_pred_checked, out_dir=out_subdir)
            print(self._fmt_secs(time.time() - t0))

            t0 = time.time()
            print('Creando matrices TRANSBORDO...')
            rutas_csv_path = Path(__file__).resolve().parent / "check_transbordo" / "rutas_transbordo.csv"
            if not rutas_csv_path.exists():
                crear_csv_rutas_transbordo(out_path=rutas_csv_path)
            crear_matriz_transbordo_macro(
                self.x_pred_checked,
                self.transbordos_implicitos,
                top_n=20,
                out_dir=out_subdir,
            )
            crear_matriz_transbordo_micro(
                self.x_pred_checked,
                self.transbordos_implicitos,
                rutas_csv_path=rutas_csv_path,
                out_dir=out_subdir,
            )
            print(self._fmt_secs(time.time() - t0))

        t0 = time.time()
        print('Calculando accuracies...')
        acc = Accuracy(hiperparams=self.hiperparams,distancia_entre_paradas = 1)
        
        exacto,aprox, dfs = acc.get_accuracy_bajadas(self.x_pred_checked, self.x_test, self.y_test)
        x_mal, y_mal, x_test, x_checked, x_checked_f,y_test,x_test_f,y_test_f,x_bajadas,y_bajadas,num_test_total,num_test_cercanias,num_test_metro,num_test_ml = dfs 

        accuracy_exacto, accuracy_exacto_cercanias,accuracy_exacto_metro,accuracy_exacto_ml = exacto
        accuracy_aprox,accuracy_aprox_cercanias,accuracy_aprox_metro,accuracy_aprox_ml = aprox
        
        print(self._fmt_secs(time.time() - t0))
    

        print(f"TOTAL: {self._fmt_secs(time.time() - t_total0)}")

        num_dias_unica_subida = (self.subidas_unicas_en_dia
          .select("id_tarjeta", F.to_date("timestamp_validacion").alias("fecha"))
          .distinct()
          .count())
        
        if self.hiperparams['paper_original'] in ['munizaga', 'lee']:
            num_dias_unica_subida_sin_pred = (self.subidas_unicas_en_dia_sin_pred
              .select("id_tarjeta", F.to_date("timestamp_validacion").alias("fecha"))
              .distinct()
              .count())
            
            
            num_dias_unica_subida_con_pred = num_dias_unica_subida - num_dias_unica_subida_sin_pred
            ''' num_dias_unica_subida_con_pred = self.x_bajadas[
                (self.x_bajadas['tipo_asignacion_2sub'] == 'next_day') &
                (self.x_bajadas['prediccion_descenso'] != 'unknown')
                ].shape[0]'''
        if self.hiperparams['paper_original'] == 'lee':
            num_dias_impredecibles_lee = (self.df_no_predecible_next_day
              .select("id_tarjeta", F.to_date("timestamp_validacion").alias("fecha"))
              .distinct()
              .count())
            

        if self.hiperparams['paper_original']== 'munizaga':
            subidas_totales = self.y_pred.shape[0] + num_dias_unica_subida_sin_pred
            
        elif self.hiperparams['paper_original']== 'lee':
             subidas_totales = (self.x_1sub[self.x_1sub['prediccion_descenso'] != 'unknown'].shape[0]) + instancia.num_no_predecibles_lee + instancia.num_unknown_lee
            
        else:
            subidas_totales = self.y_pred.shape[0] + num_dias_unica_subida
            num_dias_unica_subida_sin_pred = num_dias_unica_subida

        if self.hiperparams['paper_original']== 'lee':
            prop_desc_calculado = (self.x_1sub[self.x_1sub['prediccion_descenso'] != 'unknown'].shape[0]) / subidas_totales
        else:
            prop_desc_calculado = (self.y_pred.shape[0] - self.unknown.shape[0]) / subidas_totales

        
        print(f"""
        proporcion de subidas unicas en el dia: {num_dias_unica_subida /subidas_totales}
        proporcion de subidas con descenso unknown: {self.unknown.shape[0]/subidas_totales}
        proporcion de subidas a las que calculamos descenso: {prop_desc_calculado}"""
       ) 
        if self.hiperparams['paper_original'] in ['munizaga','lee']:
            print(f"proporcion de subidas que no se les pudo encontrar 2sub con el dia siguiente (sin pred): {num_dias_unica_subida_sin_pred /subidas_totales}")

        if self.hiperparams['paper_original']== 'lee':
            print(f"proporcion de impredecibles lee: {num_dias_impredecibles_lee /subidas_totales}\n")

            

            

            
            

        
        
        detalles_hipotesis = {
            1: '1.en prediccion metro y cercanias se prioriza la linea original\n',
            3: '3.radio_busqueda ajustado por tipo perfil\n',

            5: '5.spark\n',
            6: '6.filtro sentido de viaje metro+cercanias (producto escalar B-A · C-A >= 0)\n',
            7: 'filtrar candidatas fuera de la zona tarifaria del titulo\n',
            8: 'distancia andando con OSM\n',
            9: 'distancia_m + OSM ponderado\n',
            10: 'si distancia ajustada priorizo linea',
        }
            
        hipotesis_aplicadas='0.construccion de conjunto de validacion\n'
        for i in range(1,10):
            if i in self.hipotesis:
                hipotesis_aplicadas += ' -> '+detalles_hipotesis[i] 
        hipotesis_aplicadas += '\n'
        
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)

        try: 
            a= self.x_pred_checked[self.x_pred_checked['tipo']=='TRANSBORDO'].shape[0]/self.x_pred_checked.shape[0]
            b= self.x_pred_checked[self.x_pred_checked['tipo'] == 'BAJADA'].shape[0]/self.x_pred_checked.shape[0]
        except: 
            a= 'no calculado'
            b= 'no calculado'

        
        
        
        resumen = (
            f"resultados aplicando la metodologia de cubo izquierdo con {self.path}\n"
            f"TOTAL TIEMPO EJECUCIÓN: {self._fmt_secs(time.time() - t_total0)}\n\n"
            
            f"num total de validaciones de bajada (no usaremos todo): {self.y_test.shape[0]}\n"

            f"proporcion subidas con descenso unknown: {self.unknown.shape[0] / subidas_totales}\n"
            f"proporcion subidas a las que calculamos descenso: {prop_desc_calculado}\n\n"
            f"Tras check_tranbordo, obtenemos x_pred_checked: \n"
            
            f"proporcion de preds tipo TRANSBORDO:{a}\n"
            f"proporcion de preds tipo BAJADA: {b}\n\n"

            f"########### --TAMAÑOS CONJUNTO DE VALIDACION FINAL (validamos con {self.hiperparams['tipo_validacion']})-- ###########\n\n"

            f'x_test shape: {x_test.shape}\n'
            f'x_checked shape: {x_checked.shape}\n'
            f'x_checked_f shape: {x_checked_f.shape}\n'
            f'y_test shape: {y_test.shape}\n'
            f'y_test_f shape: {y_test_f.shape}\n'
            f'x_bajadas shape: {x_bajadas.shape}\n'
            f'y_bajadas shape: {y_bajadas.shape}\n'
            
            f'Número de filas que tiene el conjunto de validacion final:{x_bajadas.shape[0]}\n'
            f'Número de filas x_test que no estaban en x_checked (no se ha podido calcular o se quito en el procesado):{x_test.shape[0] - x_test_f.shape[0]}\n\n'

            f"########### -- ACCURACIES EXACTOS-- ###########\n\n"

            f'accuracy total: {accuracy_exacto}\n'
            f'Proporcion de cercanías en test: {num_test_cercanias/num_test_total}.\nAccuracy en cercanias: {accuracy_exacto_cercanias}\n'
            f'Proporcion de metro en test: {num_test_metro/num_test_total}.\nAccuracy en metro: {accuracy_exacto_metro}\n'
            f'Proporcion de metro ligero en test: {num_test_ml/num_test_total}.\nAccuracy en metro ligero: {accuracy_exacto_ml}\n\n'

            f"########### -- ACCURACIES APROXIMADOS (x paradas)-- ###########\n\n"

            f'Proporción de errores totales que resultan aciertos con la aproximación: {accuracy_aprox}\n'
            f'accuracy aproximado total: {accuracy_exacto+accuracy_aprox*(1-accuracy_exacto)}\n\n'
            
            f'Proporción de errores en cercanias que se vuelven acierto con la aproximación: {accuracy_aprox_cercanias}\n'
            f'accuracy aproximado cercanías: {accuracy_exacto_cercanias+accuracy_aprox_cercanias*(1-accuracy_exacto_cercanias)}\n\n'

            f'Proporción de errores en metro que se vuelven acierto con la aproximación: {accuracy_aprox_metro}\n'
            f'accuracy aproximado metro: {accuracy_exacto_metro+accuracy_aprox_metro*(1-accuracy_exacto_metro)}\n\n'

            f'Proporción de errores en metro ligero que se vuelven acierto con la aproximación: {accuracy_aprox_ml}\n'
            f'accuracy aproximado metro ligero: {accuracy_exacto_ml+accuracy_aprox_ml*(1-accuracy_exacto_ml)}\n\n'
             
            f"hipotesis probadas:\n\n {hipotesis_aplicadas}"

            f"lista de hiperapametros (tal vez alguno no es usado):\n\n {self.hiperparams}\n\n"

            f"VALOR DE LA FUNCION DE COSTE: {prop_desc_calculado * (1 - accuracy_exacto)+ 0.3 * (1 - prop_desc_calculado)}"
        )

        if self.name_out == None:
            _fname_stem = f"resultados_cubo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            _fname_stem = self.name_out

        if 12 in self.hipotesis:

            out_subdir.mkdir(parents=True, exist_ok=True)
            fname = out_subdir / f"{_fname_stem}.txt"
            fname.write_text(resumen, encoding="utf-8")
            print(f"Resumen guardado en: {fname}")
        elif self.write_results:
            fname = out_dir / f"{_fname_stem}.txt"
            fname.write_text(resumen, encoding="utf-8")
            print(f"Resumen guardado en: {fname}")

        if 3 in self.hipotesis:
            coste = calcular_coste_por_perfil(self.x_1sub,x_mal, x_bajadas,self.unknown,subidas_totales)
        else:
            coste = prop_desc_calculado * (1 - accuracy_exacto)+ 0.3 * (1 - prop_desc_calculado)
        print(f"VALOR DE LA FUNCION DE COSTE: {coste}\n")
        print(f"accuracy exacto: {accuracy_exacto}\n")
        print(f"proporcion de descensos calculados: {prop_desc_calculado}\n")

        if self.hiperparams['paper_original']== 'lee':
            num_entradas_lee = instancia.num_df_lee
            num_predecibles = num_entradas_lee - instancia.num_no_predecibles_lee
            print(f"proporcion de impredecibles lee: {instancia.num_no_predecibles_lee/ subidas_totales}\n")
            print(f"proporcion de predecibles lee entre el total de subidas: {num_predecibles/ subidas_totales}\n")
            print(f"proporcion de unknown entre las predicciones de lee: {instancia.num_unknown_lee /num_predecibles}\n")

            num_fallos_pred_lee = x_mal[x_mal['tipo_asignacion_2sub'] == 'lee'].shape[0]
            num_preds_lee = x_bajadas[x_bajadas['tipo_asignacion_2sub'] == 'lee'].shape[0]
            acc_pred_2sub_lee = (num_preds_lee - num_fallos_pred_lee)/num_preds_lee
            print(f"accuracy entre las predicciones lee: {acc_pred_2sub_lee}\n")

            print(f'proporcion de continuous travel: {(self.x_1sub[(self.x_1sub['prediccion_descenso'] != 'unknown') & (self.x_1sub['tipo_asignacion_2sub'] != 'lee')].shape[0])/subidas_totales}')
            
            print(f'proporcion de non-continuous recurrent travel: {(self.x_1sub[(self.x_1sub['prediccion_descenso'] != 'unknown') & (self.x_1sub['tipo_asignacion_2sub'] == 'lee')].shape[0])/subidas_totales}')

            print(f'resto: {(instancia.num_no_predecibles_lee + instancia.num_unknown_lee)/subidas_totales}')
            
            
        if self.hiperparams['paper_original'] in ['munizaga','lee'] :
            
            print(f"proporcion de validaciones a las que SI se le puede buscar 2a sub con el dia siguiente:{num_dias_unica_subida_con_pred /subidas_totales}\n")

            print(f'proporcion de predecibles con next day sobre el total de subidas unicas en el dia: {num_dias_unica_subida_con_pred/num_dias_unica_subida }')
            print(f"proporcion de validaciones a las que no se le puede buscar 2a sub con el dia siguiente:{num_dias_unica_subida_sin_pred /subidas_totales}\n")
            
            num_preds_2sub_next_day = x_bajadas[x_bajadas['tipo_asignacion_2sub'] == 'next_day'].shape[0]
            num_fallos_pred_2sub_next_day = x_mal[x_mal['tipo_asignacion_2sub'] == 'next_day'].shape[0]
            acc_pred_2sub_next_day = (num_preds_2sub_next_day - num_fallos_pred_2sub_next_day)/num_preds_2sub_next_day
            
            print(f"accuracy entre las predicciones cuya 2sub la cogimos del 1er boarding del next day: {acc_pred_2sub_next_day}\n")

            num = self.x_1sub[
                (self.x_1sub['tipo_asignacion_2sub'] == 'next_day') &
                (self.x_1sub['prediccion_descenso'] == 'unknown')
                ].shape[0] 
            denom = self.x_1sub[self.x_1sub['tipo_asignacion_2sub'] == 'next_day'].shape[0]
            unknown_pred_2sub_next_day = num / denom
            print(f"predicciones unknown entre preds cuya 2sub la cogimos del 1er boarding del next day: {unknown_pred_2sub_next_day}\n")

        print(self.hiperparams)

        self.almacen['coste'] = coste
        self.almacen['accuracy_exacto'] = accuracy_exacto
        self.almacen['prop_desc_calculado'] = prop_desc_calculado
        self.almacen['accuracy_exacto_cercanias'] = accuracy_exacto_cercanias
        self.almacen['accuracy_exacto_metro'] = accuracy_exacto_metro
        self.almacen['accuracy_exacto_ml'] = accuracy_exacto_ml

        if 20 in self.hipotesis or 21 in self.hipotesis:
            _almacen_dir = Path(__file__).resolve().parent.parent / "MDLF" / "almacen"
            _nombre_crudo = Path(self.path).stem
            _save_dir = _almacen_dir / _nombre_crudo
            _save_dir.mkdir(parents=True, exist_ok=True)

            self.x_pred_checked.to_parquet(_save_dir / "x_pred_checked.parquet", index=False)
            self.unknown.to_parquet(_save_dir / "unknown.parquet", index=False)
            self.x_test.to_parquet(_save_dir / "x_test.parquet", index=False)
            self.y_test.to_parquet(_save_dir / "y_test.parquet", index=False)

            _subidas_sin_pred_21 = None
            if self.hiperparams.get('paper_original') == 'munizaga':
                _subidas_sin_pred_21 = self.subidas_unicas_en_dia_sin_pred.toPandas()
            if _subidas_sin_pred_21 is not None:
                _subidas_sin_pred_21.to_parquet(_save_dir / "subidas_sin_pred.parquet", index=False)
            else:

                pd.DataFrame().to_parquet(_save_dir / "subidas_sin_pred.parquet", index=False)

            hip_label = 20 if 20 in self.hipotesis else 21
            print(f"Hipótesis {hip_label}: inputs MDLF guardados en {_save_dir}")

        return x_mal,y_mal,self.almacen

        
        

    

        
        