import pandas as pd
import pickle
import networkx as nx
import numpy as np
import ipdb
from pathlib import Path

_NEAR_DIR = Path(__file__).resolve().parent / "near"

class Accuracy():

    prefijo_medio_transporte = {
    "02": "metro",
    "A9": "metro",
    "04": "cercanias",
    "AD": "metro_ligero_tranvia",
    "AF": "metro_ligero_tranvia",
    "B1": "metro_ligero_tranvia",
}

    near_stations_cercanias = pd.read_csv(_NEAR_DIR / "near_stations_cercanias.csv")
    near_stations_metro = pd.read_csv(_NEAR_DIR / "near_stations_metro.csv")
    near_stations_ml = pd.read_csv(_NEAR_DIR / "near_stations_ml.csv")

    near_stations = pd.read_csv(_NEAR_DIR / "near_stations.csv")

    def __init__(self,hiperparams,distancia_entre_paradas = 1):
            
        self.distancia_entre_paradas = distancia_entre_paradas
        self.hiperparams = hiperparams

    def calcular_accuracy_exacto(self,x_bajadas,y_bajadas):
        """sirve tanto para un medio en concreto como para todos los medios juntos"""
        
        y_pred = x_bajadas["prediccion_descenso"]
    

        print(f'shape de x_bajadas {x_bajadas.shape}')
        mask_ok =y_pred.eq(y_bajadas["punto_acceso"])
        accuracy_exacto = mask_ok.mean()

        x_mal = x_bajadas.loc[~mask_ok].copy()
        y_mal = y_bajadas.loc[~mask_ok].copy()
        return accuracy_exacto, x_bajadas.shape[0],x_mal,y_mal

    def calcular_accuracy_aproximado(self, y_pred,y_real):
        
        bajadas_reales = y_real["punto_acceso"]
                 
        if not(y_real).empty:
            

            ns = Accuracy.near_stations.loc[Accuracy.near_stations["distancia"].le(self.distancia_entre_paradas), ["origen", "destino"]].drop_duplicates()

            pares = pd.DataFrame(
                {"origen": bajadas_reales.to_numpy(), "destino": y_pred.to_numpy()},
                index=y_pred.index
            )

            m = pares.reset_index().merge(ns, on=["origen", "destino"], how="left", indicator=True)

            mascara = m["_merge"].eq("both").to_numpy()
            mascara = pd.Series(mascara, index=pares.index, name="mascara")
            
            accuracy_aprox = mascara.mean()
            
            return accuracy_aprox
        else:
            print('No ha habido ningún error en este medio, por lo que el accuracy aprox y exacto es 1')
            return 1.0

    

    def particiones_por_medio(self,x_bajadas,y_bajadas):

        x_bajadas_cercanias = x_bajadas[x_bajadas['tipo_transporte'] == 'cercanias']
        y_bajadas_cercanias = y_bajadas[x_bajadas['tipo_transporte'] == 'cercanias']
        
        x_bajadas_metro = x_bajadas[x_bajadas['tipo_transporte'] == 'metro']
        y_bajadas_metro = y_bajadas[x_bajadas['tipo_transporte'] == 'metro']
        
        x_bajadas_ml = x_bajadas[x_bajadas['tipo_transporte'] == 'metro_ligero_tranvia']
        y_bajadas_ml = y_bajadas[x_bajadas['tipo_transporte'] == 'metro_ligero_tranvia']

        return x_bajadas_cercanias, y_bajadas_cercanias,x_bajadas_metro,y_bajadas_metro, x_bajadas_ml,y_bajadas_ml

    def eliminar_incoherencia_simultanea_metro_bus(self,x_checked):
        d = x_checked.copy()
        key_cols = ["id_tarjeta", "timestamp_validacion"]
        dup_mask = d.duplicated(subset=key_cols, keep=False)
    
        conflict_set = {"metro", "metro_ligero_tranvia", "cercanias"}
    
        def es_conflictivo(grp: pd.DataFrame) -> bool:
            medios = set(grp["tipo_transporte"].dropna().unique())
            return ("bus_interurbano" in medios) and (len(medios & conflict_set) > 0)
    
        conflict_groups = (
            d.loc[dup_mask, key_cols + ["tipo_transporte"]]
             .groupby(key_cols, sort=False)
             .filter(es_conflictivo)[key_cols]
             .drop_duplicates()
        )
    
        conflict_index = pd.MultiIndex.from_frame(conflict_groups)
        df_index = pd.MultiIndex.from_frame(d[key_cols])
    
        inconsistente_mask = df_index.isin(conflict_index)
    
        limpio = d.loc[~inconsistente_mask].reset_index(drop=True)
        inconsistencias = d.loc[inconsistente_mask].reset_index(drop=True)
    
        return limpio, inconsistencias
        

    def definir_conjunto_validacion_final(self, x_checked: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):

        """Accuracy solo en filas tipo == 'BAJADA', pero además exigimos que cada (id_tarjeta, timestamp_validacion)
        de x_checked esté presente en x_test y que cada fila de x_test esté presente en x_checked.
        """

        key_cols = ["id_tarjeta", "timestamp_validacion"]

        print("dups x_test:", x_test.duplicated(key_cols).sum())
        print("dups x_checked:", x_checked.duplicated(key_cols).sum())

        x_checked["timestamp_validacion"] = pd.to_datetime(x_checked["timestamp_validacion"], utc=True, errors="coerce")

        x_checked,inconsistencias = self.eliminar_incoherencia_simultanea_metro_bus(x_checked)

        
        dup_mask = x_checked.duplicated(subset=key_cols, keep=False)
        dups_x_checked = (
            x_test.loc[dup_mask]
            .sort_values(key_cols)
        )
        

        

        for df in (x_test, x_checked, y_test):
            df["timestamp_validacion"] = pd.to_datetime(df["timestamp_validacion"], utc=True, errors="coerce")
        

        xt_keys = pd.MultiIndex.from_frame(x_test[key_cols])
        xc_keys = pd.MultiIndex.from_frame(x_checked[key_cols])
        

        mask_xc_in_xt = xc_keys.isin(xt_keys)
        idx_checked_en_xtest = x_checked.index[mask_xc_in_xt]
        
        x_checked_f = x_checked.loc[idx_checked_en_xtest].copy()
        xc_f_keys = pd.MultiIndex.from_frame(x_checked_f[key_cols])

        

        
        mask_xt_in_xc = xt_keys.isin(xc_f_keys)
        
        x_test_f = x_test.loc[mask_xt_in_xc].copy()
        y_test_f = y_test.loc[mask_xt_in_xc].copy()
        

        x_checked_f = x_checked_f.sort_values(key_cols).reset_index(drop=True)
        sort_idx    = x_test_f.sort_values(key_cols).index
        x_test_f    = x_test_f.loc[sort_idx].reset_index(drop=True)
        y_test_f    = y_test_f.loc[sort_idx].reset_index(drop=True)

        print("len x_checked_f:", len(x_checked_f), "len y_test_f:", len(y_test_f))

        
        if self.hiperparams['tipo_validacion'] == 'TODO':
            mask_bajadas = x_checked_f["prediccion_descenso"].ne("unknown")
        elif self.hiperparams['tipo_validacion'] == 'TRANSBORDO':
            mask_bajadas = x_checked_f["tipo"].eq("TRANSBORDO")
        else:
            mask_bajadas = x_checked_f["tipo"].eq("BAJADA")
            
        x_bajadas = x_checked_f.loc[mask_bajadas].copy()
        print('proporcion de perfiles x_bajadas: ',x_bajadas["cod_perfil"].value_counts(normalize=True))

        y_bajadas = y_test_f.iloc[mask_bajadas.to_numpy()].copy()

        print(f'x_test shape: {x_test.shape}')
        print(f'x_checked shape: {x_checked.shape}')
        print(f'x_checked_f shape: {x_checked_f.shape}')
        print(f'y_test shape: {y_test.shape}')
        print(f'y_test_f shape: {y_test_f.shape}')
        print(f'x_bajadas shape: {x_bajadas.shape}')
        print(f'y_bajadas shape: {y_bajadas.shape}')
        
        
        print(f'Número de filas de bajada para la validacion:{x_bajadas.shape[0]}')
        print(f'Número de filas x_test que no estaban en x_checked (no se ha podido calcular o se quito en el procesado):{x_test.shape[0] - x_test_f.shape[0]}')
        
        return x_checked_f,x_test_f,y_test_f,x_bajadas,y_bajadas

    def get_accuracy_bajadas(self, x_checked: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
        
        
        x_checked_f,x_test_f,y_test_f,x_bajadas,y_bajadas = self.definir_conjunto_validacion_final(x_checked, x_test,y_test)

        x_bajadas_cercanias, y_bajadas_cercanias,x_bajadas_metro,y_bajadas_metro, x_bajadas_ml,y_bajadas_ml = self.particiones_por_medio(x_bajadas, y_bajadas)

        accuracy_exacto, num_test_total, x_mal, y_mal = self.calcular_accuracy_exacto(x_bajadas,y_bajadas)
        print(f'accuracy total: {accuracy_exacto}')

        accuracy_exacto_cercanias, num_test_cercanias,x_mal_cercanias,y_mal_cercanias = self.calcular_accuracy_exacto(
            x_bajadas_cercanias,y_bajadas_cercanias)
        print(f'Proporcion de cercanías en test: {num_test_cercanias/num_test_total}.Accuracy en cercanias: {accuracy_exacto_cercanias}')
        
        accuracy_exacto_metro,num_test_metro,x_mal_metro,y_mal_metro = self.calcular_accuracy_exacto(x_bajadas_metro,y_bajadas_metro)
        print(f'Proporcion de metro en test: {num_test_metro/num_test_total}.Accuracy en metro: {accuracy_exacto_metro}')
        
        accuracy_exacto_ml,num_test_ml,x_mal_ml,y_mal_ml = self.calcular_accuracy_exacto(x_bajadas_ml,y_bajadas_ml)
        print(f'Proporcion de metro ligero en test: {num_test_ml/num_test_total}.Accuracy en metro ligero: {accuracy_exacto_ml}')

        print('ACCURACY APROXIMADO: \n')

        accuracy_aprox = self.calcular_accuracy_aproximado(x_mal["prediccion_descenso"],y_mal)
        print(f'Proporción de errores totales que resultan aciertos con la aproximación: {accuracy_aprox}')
        print(f'accuracy aproximado total: {accuracy_exacto+accuracy_aprox*(1-accuracy_exacto)}\n')

        accuracy_aprox_cercanias = self.calcular_accuracy_aproximado(x_mal_cercanias["prediccion_descenso"],y_mal_cercanias)
        print(f'Proporción de errores en cercanias que se vuelven acierto con la aproximación: {accuracy_aprox_cercanias}')
        print(f'accuracy aproximado cercanías: {accuracy_exacto_cercanias+accuracy_aprox_cercanias*(1-accuracy_exacto_cercanias)}\n')

        accuracy_aprox_metro = self.calcular_accuracy_aproximado(x_mal_metro["prediccion_descenso"],y_mal_metro)
        print(f'Proporción de errores en metro que se vuelven acierto con la aproximación: {accuracy_aprox_metro}')
        print(f'accuracy aproximado metro: {accuracy_exacto_metro+accuracy_aprox_metro*(1-accuracy_exacto_metro)}\n')

        accuracy_aprox_ml = self.calcular_accuracy_aproximado(x_mal_ml["prediccion_descenso"],y_mal_ml)
        print(f'Proporción de errores en metro ligero que se vuelven acierto con la aproximación: {accuracy_aprox_ml}')
        print(f'accuracy aproximado metro ligero: {accuracy_exacto_ml+accuracy_aprox_ml*(1-accuracy_exacto_ml)}\n')
        

        
               
        exacto = (accuracy_exacto, accuracy_exacto_cercanias,accuracy_exacto_metro,accuracy_exacto_ml)
        aprox = (accuracy_aprox,accuracy_aprox_cercanias,accuracy_aprox_metro,accuracy_aprox_ml)

        dfs = (x_mal, y_mal, x_test, x_checked, x_checked_f,y_test,x_test_f,y_test_f,x_bajadas,y_bajadas,num_test_total,num_test_cercanias,num_test_metro,num_test_ml)
        return exacto, aprox,dfs 

        

    def dist_entre_bajada_y_2sub(self,lat_bajada,lon_bajada, lat_2a_sub,lon_2a_sub):
        '''input 2 series de pandas,output serie de pandas con la distancia haversine calculada'''
        R = 6371000.0

        lat1 = np.radians(lat_bajada.astype(float))
        lon1 = np.radians(lon_bajada.astype(float))
        lat2 = np.radians(lat_2a_sub.astype(float))
        lon2 = np.radians(lon_2a_sub.astype(float))
    

        dlat = lat2 - lat1
        dlon = lon2 - lon1
    
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        distancias = R * c
    

        return pd.Series(distancias, index=lat_bajada.index)
        

        

                
                

