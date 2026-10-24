import pandas as pd

import pandas as pd
from show_side_by_side import *

def crear_mapeos(validaciones: pd.DataFrame,hiperparams):
    """
    Devuelve (x_test_df, y_test_df) donde:
    - y_test son filas con id_resultado en  codigos descensos)
    - x_test es la fila anterior a cada una de esas (subida asociada)
    """

    prefijo_medio_transporte = {
    "02": "metro",
    "A9": "metro",
    "03": "bus_emt",
    "04": "cercanias",
    "AD": "metro_ligero_tranvia",
    "AF": "metro_ligero_tranvia",
    "B1": "metro_ligero_tranvia",
    "1A": "bus_interurbano",
    "1B": "bus_interurbano",
    "20": "bus_interurbano",
    "23": "bus_interurbano",
    "24": "bus_interurbano",
    "25": "bus_interurbano",
    "26": "bus_interurbano",
    "2C": "bus_interurbano",
    "2E": "bus_interurbano",
    "2F": "bus_interurbano",
    "30": "bus_interurbano",
    "31": "bus_interurbano",
    "33": "bus_interurbano",
    "34": "bus_interurbano",
    "35": "bus_interurbano",
    "38": "bus_interurbano",
    "39": "bus_interurbano",
    "3A": "bus_interurbano",
    "3B": "bus_interurbano",
    "3C": "bus_interurbano",
    "3D": "bus_interurbano",
    "43": "bus_interurbano",
    "45": "bus_interurbano",
    "47": "bus_interurbano",
    "4A": "bus_interurbano",
    "90": "bus_interurbano",
}
    codigos_descenso = ('9', '44', '45','81', '79', '34','73','75','77','10')
    if validaciones.empty:
        return validaciones.copy(), validaciones.copy()

    idx_x, idx_y = [], []

    prev_idx = validaciones.index[0]
    prev_id_res =validaciones.at[prev_idx,'id_resultado']
    
    for idx in validaciones.index[1:]:
        id_res = validaciones.at[idx, "id_resultado"]

        
        if (id_res in codigos_descenso) and (prev_id_res not in codigos_descenso) and (validaciones.at[idx,'id_tarjeta']==validaciones.at[prev_idx,'id_tarjeta']):

            pac1 = validaciones.at[idx, "punto_acceso"]
            pac2 = validaciones.at[prev_idx, "punto_acceso"]
            
            medio1 = prefijo_medio_transporte[pac1[:2]]
            medio2 = prefijo_medio_transporte[pac2[:2]]

            if medio1 not in ('metro', 'cercanias','metro_ligero_tranvia') or ((pac1 == pac2) and (hiperparams['paper_original'] != 'cubo')):

                prev_idx = idx
                prev_id_res = id_res
                continue
                

            ts1 =  pd.to_datetime(validaciones.at[idx, "timestamp_validacion"])
            ts2 = pd.to_datetime(validaciones.at[prev_idx, "timestamp_validacion"])
            fecha1 = validaciones.at[idx, "fecha"]
            fecha2 = validaciones.at[prev_idx, "fecha"]
            dif_temporal = ts1 - ts2
            

            if ((medio1 == medio2) 
                and ((fecha1 == fecha2) or (hiperparams['paper_original'] == 'cubo'))
                and (
                (
                    medio1 == "cercanias"
                    and dif_temporal <= pd.Timedelta(hours=2, minutes=30)
                )
                or (
                    medio1 in ("metro", "metro_ligero_tranvia")
                    and dif_temporal <= pd.Timedelta(hours=1, minutes=15))
                or (hiperparams['paper_original'] == 'cubo')
            )):
            
        
                idx_x.append(prev_idx)
                idx_y.append(idx)
                
                

            elif (
            ( abs(
                 pd.to_datetime(validaciones.at[idx, "timestamp_validacion"]) -
                         pd.to_datetime(validaciones.at[prev_idx, "timestamp_validacion"])
                       ).total_seconds() <= 10) 
            and (fecha1 == fecha2) 
            and (hiperparams['paper_original'] != 'cubo')):
                medio2 = prefijo_medio_transporte[validaciones.at[prev_idx-1, "punto_acceso"][:2]]
                fecha2 = pd.to_datetime(validaciones.at[prev_idx-1, "timestamp_validacion"]).date()

                if (medio1 == medio2) and (fecha1 == fecha2):
                    idx_x.append(prev_idx-1)
                    idx_y.append(idx)
                    
    
        prev_idx = idx
        prev_id_res = id_res

    x_test_df = validaciones.loc[idx_x].copy().reset_index(drop=True)
    y_test_df = validaciones.loc[idx_y].copy().reset_index(drop=True)

    if hiperparams['paper_original'] != 'cubo':
        mask_barrio_puerto = (
        x_test_df["nombre_parada"].eq('BARRIO DEL PUERTO')
        )
    
        x_test_df = x_test_df.loc[~mask_barrio_puerto].reset_index(drop=True)
        y_test_df = y_test_df.loc[~mask_barrio_puerto].reset_index(drop=True)
    

    y_test_df.loc[y_test_df["punto_acceso"] == "02_L12_P22", "punto_acceso"] = "02_L3_P58"

    
    
    return x_test_df, y_test_df

        

