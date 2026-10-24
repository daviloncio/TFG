import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from check_transbordo.check_transbordo import CheckTransbordo

BASE_DIR = Path(__file__).resolve().parent

class PrediccionDescenso:
    def __init__(self, lote_inicial: pd.DataFrame, hipotesis={}):
        self.fichero_650 = pd.read_csv(
            BASE_DIR / ".." / ".." / "topologia" / "paradas_vecinas_650m_03_02.csv"
        )

        buffer = pd.read_csv(BASE_DIR / "buffer_paradas.csv")
        self.buffer_map = dict(zip(buffer["id_subidas"], buffer["prediccion"]))
        self.buffer_dirty = False
        self.hipotesis = hipotesis

        dtype_route_id = {"route_id": "string"}
        dtype_pacs_inter = {
            "DPAYPOINT": "string",
            "IDFLINEAGESTRA": "string",
            "IDFPARADAGESTRA": "string",
        }

        self.rutas_inter = pd.read_csv(
            BASE_DIR / "rutas_interurbano_urbano.csv",
            dtype=dtype_route_id,
        )
        self.pacs_inter = pd.read_csv(
            BASE_DIR / "pac_interurbanos.csv",
            dtype=dtype_pacs_inter,
        )

        dtype_pacs_emt = {
            "DPAYPOINT": "string",
            "IDFLINEAGESTRA": "string",
            "CODIGOPARADAEMT": "string",
        }

        self.rutas_emt = pd.read_csv(
            BASE_DIR / "rutas_EMT.csv",
            dtype=dtype_route_id,
        )
        self.rutas_emt["route_id"] = (
            self.rutas_emt["route_id"]
            .astype("string")
            .str.replace(r"^0+(?=\d)", "", regex=True)
        )

        self.pacs_emt = pd.read_csv(
            BASE_DIR / "topologia_solo_EMT.csv",
            dtype=dtype_pacs_emt,
        )

        dtype_pacs_ml = {
            "DPAYPOINT": "string",
            "IDFLINEAGESTRA": "string",
            "IDFESTACION": "string",
        }

        self.rutas_ml = pd.read_csv(
            BASE_DIR / "rutas_ml.csv",
            dtype=dtype_route_id,
        )
        self.pacs_ml = pd.read_csv(
            BASE_DIR / "pacs_ml.csv",
            dtype=dtype_pacs_ml,
        )

        self.pacs_trenes = pd.read_csv(BASE_DIR / "topologia_trenes2.csv")
        self.check = CheckTransbordo()

        lote_inicial = lote_inicial.copy()
        lote_inicial["prediccion_descenso"] = "unknown"
        lote_inicial["fecha"] = pd.to_datetime(
            lote_inicial["timestamp_validacion"]
        ).dt.date

        mask_unlinked = (
            lote_inicial.groupby("fecha")["fecha"].transform("size") == 1
        )

        self.df_unlinked = lote_inicial[mask_unlinked].copy()
        self.lote_inicial = lote_inicial
        self.mask_unlinked = mask_unlinked

    def construir_lotes(self) -> None:
        lote = self.lote_inicial[~self.mask_unlinked].copy()
        self.lote = lote

        first_idx_by_day = (
            lote.groupby(["id_tarjeta", "fecha"], sort=False)
            .head(1)
            .index
            .to_series()
        )

        first_idx_by_day = (
            first_idx_by_day.groupby(
                [
                    lote.loc[first_idx_by_day.index, "id_tarjeta"],
                    lote.loc[first_idx_by_day.index, "fecha"],
                ]
            )
            .first()
            .to_dict()
        )

        self.first_idx_by_day = first_idx_by_day
        self.lote_primera_sub = lote.copy()

        if lote.empty:
            self.lote_inicial = self.lote_inicial.copy()
            self.lote_inicial["tipo"] = "UNLINKED"
            print(
                "WARNING!! TODAS LAS VALIDACIONES PASADAS SON UNLINKED YA QUE NO HAN CONSEGUIDO UNA PREDICCIÓN EN EL PASO INTERIOR. \n"
                "            NO EJECUTAMOS LA DETECCION DE TRANSBORDO"
            )
            self.todo_unlinked = True
            return

        self.todo_unlinked = False

        idxs = lote.index.to_list()
        sub2_idxs = []

        for pos, idx1 in enumerate(idxs):
            sub1 = lote.loc[idx1]

            if pos + 1 < len(idxs):
                idx2_cand = idxs[pos + 1]
                sub2_cand = lote.loc[idx2_cand]

                if sub2_cand["id_tarjeta"] == sub1["id_tarjeta"]:
                    sub2_idxs.append(idx2_cand)
                    continue

            sub2_idxs.append(
                first_idx_by_day[(sub1["id_tarjeta"], sub1["fecha"])]
            )

        self.lote_segunda_sub = lote.loc[sub2_idxs].copy()

    def bucle_predecciones(self):
        if getattr(self, "todo_unlinked", False):

            return self.lote_inicial, None, None, self.lote_inicial.copy()

        lote = self.lote
        lote_primera_sub = self.lote_primera_sub
        lote_segunda_sub = self.lote_segunda_sub
        first_idx_by_day = self.first_idx_by_day

        i = -1

        for (idx1, sub1), (idx2, sub2) in tqdm(
            zip(lote_primera_sub.iterrows(), lote_segunda_sub.iterrows()),
            total=len(lote_primera_sub),
            desc="prediccion_descenso",
        ):
            i += 1

            if (
                sub1["id_tarjeta"] != sub2["id_tarjeta"]
                or sub1["fecha"] != sub2["fecha"]
            ):
                idx2_real = first_idx_by_day[(sub1["id_tarjeta"], sub1["fecha"])]
                fila_origen = lote.loc[idx2_real]

                if isinstance(fila_origen, pd.DataFrame):
                    fila_origen = fila_origen.iloc[0]

                sub2 = fila_origen
                self.lote_segunda_sub.iloc[i] = fila_origen[
                    self.lote_segunda_sub.columns
                ].to_numpy()

            p_acc_sub1 = sub1["punto_acceso"]
            p_acc_sub2 = sub2["punto_acceso"]
            medio_1a_sub = sub1["tipo_transporte"]

            id_subidas = f"{p_acc_sub1}_y_{p_acc_sub2}"

            if not any(x in self.hipotesis for x in {1, 2}):
                pred = self.buffer_map.get(id_subidas)
                if pred is not None and pred == pred:
                    lote.loc[idx1, "prediccion_descenso"] = pred
                    continue

            candidatas = self.fichero_650.loc[
                self.fichero_650["punto_acceso"] == p_acc_sub2
            ]

            fila_self = pd.DataFrame(
                [
                    {
                        "punto_acceso": p_acc_sub2,
                        "vecina": p_acc_sub2,
                        "tipo_transporte_vecina": sub2["tipo_transporte"],
                        "distancia_m": 0.0,
                    }
                ]
            )

            candidatas = pd.concat(
                [candidatas, fila_self],
                ignore_index=True,
            )

            if p_acc_sub1[:2] == "B1":
                sufijo = p_acc_sub1.split("_")[-1]
                l1 = f"B1_L1_{sufijo}"
                l2 = f"B1_L2_{sufijo}"
                candidatas = candidatas.loc[
                    ~candidatas["vecina"].isin([l1, l2])
                ]
            else:
                candidatas = candidatas.loc[
                    candidatas["vecina"] != p_acc_sub1
                ]

            candidatas = candidatas.loc[
                candidatas["tipo_transporte_vecina"] == medio_1a_sub
            ]

            cumple_requisitos = []

            for _, row in candidatas.iterrows():
                parada = row.get("vecina", None)

                if parada is None:
                    print(f"el p_acc {parada} no tiene vecinos")
                    continue

                if medio_1a_sub in ("metro", "cercanias"):
                    cumple_requisitos.append(parada)
                    continue

                if medio_1a_sub == "bus_interurbano":
                    fichero_rutas = self.rutas_inter
                    fichero_pacs = self.pacs_inter
                    nombre_col = "IDFPARADAGESTRA"
                    col_route_id = "route_id"
                elif medio_1a_sub == "bus_emt":
                    fichero_rutas = self.rutas_emt
                    fichero_pacs = self.pacs_emt
                    nombre_col = "CODIGOPARADAEMT"
                    col_route_id = "route_id"
                else:
                    fichero_rutas = self.rutas_ml
                    fichero_pacs = self.pacs_ml
                    nombre_col = "IDFESTACION"
                    col_route_id = "IDFLINEAGESTRA"

                tmp1 = fichero_pacs.loc[
                    fichero_pacs["DPAYPOINT"] == p_acc_sub1,
                    ["IDFLINEAGESTRA", nombre_col],
                ]
                tmp2 = fichero_pacs.loc[
                    fichero_pacs["DPAYPOINT"] == parada,
                    ["IDFLINEAGESTRA", nombre_col],
                ]

                if tmp1.empty or tmp2.empty:
                    continue

                id_linea = tmp1.iloc[0]["IDFLINEAGESTRA"]
                id_linea_c = tmp2.iloc[0]["IDFLINEAGESTRA"]
                stop_id = tmp1.iloc[0][nombre_col]
                stop_id_c = tmp2.iloc[0][nombre_col]

                if not isinstance(id_linea, str):
                    raise TypeError(f"id_linea no es str: {type(id_linea)}")

                if id_linea != id_linea_c or stop_id == stop_id_c:
                    continue

                rutas = fichero_rutas.loc[
                    fichero_rutas[col_route_id] == id_linea,
                    "ruta_paradas",
                ]

                for ruta_str in rutas:
                    try:
                        ruta = json.loads(ruta_str)
                    except (TypeError, json.JSONDecodeError):
                        continue

                    try:
                        if ruta.index(stop_id) <= ruta.index(stop_id_c):
                            cumple_requisitos.append(parada)
                            break
                    except Exception:
                        continue

            prediccion = "unknown"
            n_finalistas = len(cumple_requisitos)

            if n_finalistas == 0:
                pass

            elif n_finalistas == 1:
                prediccion = cumple_requisitos[0]
                lote.loc[idx1, "prediccion_descenso"] = prediccion

            else:

                if 2 in self.hipotesis and medio_1a_sub in ("metro", "cercanias"):
                    finalistas_zona = {
                        p_acc
                        for p_acc in cumple_requisitos
                        if self.pacs_trenes[
                            self.pacs_trenes["DPAYPOINT"] == p_acc
                        ]["CORONATARIFARIA"].iloc[0]
                        in set(sub1["zonas_permitidas"])
                    }

                    if len(finalistas_zona) == 0:
                        raise ValueError(
                            "no entiendo pq pero hemos llegado a un problema con la zona tarifaria"
                        )

                if 1 in self.hipotesis and medio_1a_sub in ("metro", "cercanias"):
                    finalistas = {
                        p_acc: float(
                            candidatas.loc[
                                candidatas["vecina"] == p_acc, "distancia_m"
                            ].astype(float).iloc[0]
                        ) / 1.0
                        + self.check.calculo_tiempo_estimado(
                            p_acc_sub1,
                            p_acc,
                            devolver_ruta=False,
                        )
                        for p_acc in cumple_requisitos
                    }
                else:
                    finalistas = {
                        p_acc: float(
                            candidatas.loc[
                                candidatas["vecina"] == p_acc, "distancia_m"
                            ].min()
                        )
                        for p_acc in cumple_requisitos
                    }

                prediccion = min(finalistas, key=finalistas.get)
                dist_ganadora = finalistas[prediccion]
                lote.loc[idx1, "prediccion_descenso"] = prediccion

            if prediccion == "unknown":
                continue

            if any(x in self.hipotesis for x in {1, 2}):
                continue

            self.buffer_map[id_subidas] = prediccion
            self.buffer_dirty = True

        if self.buffer_dirty and not any(x in self.hipotesis for x in {1, 2}):
            buffer_out = pd.DataFrame(
                {
                    "id_subidas": list(self.buffer_map.keys()),
                    "prediccion": list(self.buffer_map.values()),
                }
            )
            buffer_out.to_csv(BASE_DIR / "buffer_paradas.csv", index=False)

        df_unknown = lote[lote["prediccion_descenso"] == "unknown"]

        print(
            "num de validaciones solitas en el dia en PRED DESCENSO.py",
            self.df_unlinked.shape[0],
        )
        print(
            "num de validaciones con pred = unknown",
            df_unknown.shape[0],
        )

        if len(self.lote_primera_sub) != len(self.lote_segunda_sub):
            raise ValueError(
                "lote_primera_sub y lote_segunda_sub no tienen la misma longitud"
            )

        self.lote_primera_sub["punto_acceso_2sub"] = (
            self.lote_segunda_sub["punto_acceso"].to_numpy()
        )
        self.lote_primera_sub["LATITUD_2sub"] = (
            self.lote_segunda_sub["LATITUD"].to_numpy()
        )
        self.lote_primera_sub["LONGITUD_2sub"] = (
            self.lote_segunda_sub["LONGITUD"].to_numpy()
        )

        return (
            self.lote_primera_sub,
            self.lote_segunda_sub,
            lote["prediccion_descenso"],
            df_unknown,
        )