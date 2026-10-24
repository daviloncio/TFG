import pickle, heapq
import networkx as nx
from typing import Dict, Set, Tuple, Optional, Any, List
import pandas as pd
import ast
from pathlib import Path
from tqdm.auto import tqdm
import ipdb
BASE_DIR = Path(__file__).resolve().parent
PRED_DIR = BASE_DIR.parent / "prediccion_descenso"

class CheckTransbordo:
    def __init__(
        self,
        cercanias_graph_path: str = BASE_DIR / "cercanias_multidigraph_sin_civis.pkl",
        metro_graph_path: str = BASE_DIR / "metro_multidigraph.pkl",
        hipotesis = {}
    ):
        self.hipotesis = hipotesis
        self.rutas_ml_con_tiempos =  pd.read_csv(BASE_DIR / 'rutas_ml_con_tiempos.csv')
        self.topologia_trenes = pd.read_csv(PRED_DIR / "topologia_trenes2.csv")

        self.rutas_interurbano_urbano_con_tiempos =  pd.read_csv(BASE_DIR / 'rutas_interurbano_urbano_con_tiempos.csv')
        self.topologia_sin_EMT = pd.read_csv( PRED_DIR /  "topologia_sin_EMT.csv")

        self.rutas_EMT_con_tiempos = pd.read_csv(BASE_DIR / 'rutas_EMT_con_tiempos.csv')
        self.rutas_EMT_con_tiempos["route_id"] = self.rutas_EMT_con_tiempos["route_id"].astype(str)

        self.topologia_solo_EMT = pd.read_csv( PRED_DIR /  "topologia_solo_EMT.csv")     
        

        self.topologia_solo_EMT["IDFLINEAGESTRA"] = (
            pd.to_numeric(self.topologia_solo_EMT["IDFLINEAGESTRA"], errors="coerce")
              .astype("Int64")
              .astype("string")
            )
        self.topologia_solo_EMT["CODIGOPARADAEMT"] = self.topologia_solo_EMT["CODIGOPARADAEMT"].astype(str)

    

        with open(cercanias_graph_path, "rb") as f:
            self.G_cercanias = pickle.load(f)
        if not isinstance(self.G_cercanias, nx.MultiDiGraph):
            raise TypeError(f"Cercanías: se esperaba nx.MultiDiGraph, pero es {type(self.G_cercanias)}")

        with open(metro_graph_path, "rb") as f:
            self.G_metro = pickle.load(f)
        if not isinstance(self.G_metro, nx.MultiDiGraph):
            raise TypeError(f"Metro: se esperaba nx.MultiDiGraph, pero es {type(self.G_metro)}")

        self.ap_to_stations_cercanias: Dict[str, Set[str]] = self._build_ap_index(self.G_cercanias)
        self.ap_to_stations_metro: Dict[str, Set[str]] = self._build_ap_index(self.G_metro)

    def _build_ap_index(self, G: nx.MultiDiGraph) -> Dict[str, Set[str]]:
        ap_to_stations: Dict[str, Set[str]] = {}
        for node, data in G.nodes(data=True):
            aps = data.get("access_point_codes", set())
            if isinstance(aps, str):
                aps = {aps}
            for ap in aps:
                ap_to_stations.setdefault(ap, set()).add(node)
        return ap_to_stations

    def _edge_line_id(self, u: str, v: str, key: Any, attrs: dict) -> str:
        return attrs.get("line_id") or attrs.get("lines") or str(key)

    def _pick_graph(self, origen_ap: str):
        """dado el punto de acceso nos fijamos en cómo empieza para inferir su medio de transporte. 
        deuvuelve el grafo o los archivos necesarios en caso de ser bus o ML"""
        
        if isinstance(origen_ap, str) and (origen_ap.startswith("02_") or origen_ap.startswith("A9_")):
            return self.G_metro, self.ap_to_stations_metro, True
        elif isinstance(origen_ap, str) and origen_ap.startswith("04_"):
            return self.G_cercanias, self.ap_to_stations_cercanias, True
            
        elif isinstance(origen_ap, str) and origen_ap.startswith("03_"):
            
            return self.rutas_EMT_con_tiempos, self.topologia_solo_EMT, False

        elif isinstance(origen_ap, str) and (
            origen_ap.startswith("AD" ) or origen_ap.startswith("AF" ) or origen_ap.startswith("B1")
        ):
            return self.rutas_ml_con_tiempos, self.topologia_trenes , False 
        else:
            return self.rutas_interurbano_urbano_con_tiempos, self.topologia_sin_EMT,False
        
    
    def _hms_to_seconds(self,x: str) -> int:
        """
        Convierte 'HH:MM:SS' a segundos.
        Soporta HH >= 24 (típico en GTFS).
        """
        h, m, s = map(int, x.split(":"))
        return h * 3600 + m * 60 + s        
            

    def calculo_tiempo_estimado(
        self,
        origen_ap: str,
        destino_ap: str,
        transbordo_min: float = 10.0,
        devolver_ruta: bool = False,
    ):
        G, ap_to_stations,use_graph = self._pick_graph(origen_ap)

        if use_graph:

            origins = ap_to_stations.get(origen_ap, set())
            dests = ap_to_stations.get(destino_ap, set())
    
            if not origins:
                raise ValueError(f"No encuentro estación con access_point_code = {origen_ap} (orgien) en el grafo seleccionado")
            if not dests:
                
                raise ValueError(f"No encuentro estación con access_point_code = {destino_ap} (destino) y {origen_ap} (orgien) en el grafo seleccionado")
    
            dist: Dict[Tuple[str, Optional[str]], float] = {}
            prev: Dict[Tuple[str, Optional[str]], Tuple[Tuple[str, Optional[str]], Tuple[str, str, Any]]] = {}
            pq: List[Tuple[float, str, Optional[str]]] = []
    
            for s in origins:
                dist[(s, None)] = 0.0
                heapq.heappush(pq, (0.0, s, None))
    
            best_goal_state: Optional[Tuple[str, Optional[str]]] = None
    
            while pq:
                cost, u, last_line = heapq.heappop(pq)
                state_u = (u, last_line)
                if cost != dist.get(state_u, float("inf")):
                    continue
    
                if u in dests:
                    best_goal_state = state_u
                    break
    
                for _, v, key, attrs in G.out_edges(u, keys=True, data=True):
                    w = attrs.get("weight", 0)
                    if w == 0:
                        continue
    
                    line = self._edge_line_id(u, v, key, attrs)
                    penalty = float(transbordo_min) if (last_line is not None and line != last_line) else 0.0
    
                    new_cost = cost + float(w) + penalty
                    state_v = (v, line)
    
                    if new_cost < dist.get(state_v, float("inf")):
                        dist[state_v] = new_cost
                        prev[state_v] = (state_u, (u, v, key))
                        heapq.heappush(pq, (new_cost, v, line))
    
            if best_goal_state is None:
                raise ValueError(f"No hay camino entre {origen_ap} -> {destino_ap} en el grafo seleccionado")
    
            best_time = dist[best_goal_state]
    

    
            stations: List[str] = []
            lines: List[Optional[str]] = []
            cur = best_goal_state
            while cur in prev:
                pstate, (u, v, key) = prev[cur]
                stations.append(cur[0])
                lines.append(cur[1])
                cur = pstate
    
            stations.append(cur[0])
            stations.reverse()
            lines.reverse()
            if devolver_ruta:

                ruta_aps = []
                for st in stations:
                    aps = G.nodes[st].get("access_point_codes", set())
                    if isinstance(aps, str):
                        aps = {aps}
                    ruta_aps.extend(sorted(aps))
                return best_time  * 60, ruta_aps
            else:
                return best_time  * 60

        else:
            rutas_y_tiempos = G

            
            if rutas_y_tiempos is not self.rutas_ml_con_tiempos:
                
                
                
                if rutas_y_tiempos is self.rutas_EMT_con_tiempos:
                    origen_linea,origen_stop = self.topologia_solo_EMT.loc[
                        self.topologia_solo_EMT["DPAYPOINT"] == origen_ap, ["NULINGES",'CODIGOPARADAEMT']
                    ].iloc[0]
                    origen_linea = str(origen_linea)

                    destino_linea, destino_stop = self.topologia_solo_EMT.loc[
                        self.topologia_solo_EMT["DPAYPOINT"] == destino_ap, ["NULINGES",'CODIGOPARADAEMT']
                    ].iloc[0]
                    destino_linea = str(destino_linea)
                    
                else:
                    origen_linea,origen_stop = self.topologia_sin_EMT.loc[
                        self.topologia_sin_EMT["DPAYPOINT"] == origen_ap, ["IDFLINEAGESTRA",'IDFPARADAGESTRA']
                    ].iloc[0]

                    destino_linea, destino_stop = self.topologia_sin_EMT.loc[
                        self.topologia_sin_EMT["DPAYPOINT"] == destino_ap, ["IDFLINEAGESTRA",'IDFPARADAGESTRA']
                    ].iloc[0]
                
                    
                    
                assert origen_linea == destino_linea, 'orgien y destino no pertenecen a la misma linea'

                
                
                itinerarios = rutas_y_tiempos.loc[rutas_y_tiempos['route_id'] == origen_linea]
                
                
                
            else:
                
                origen_stop = self.topologia_trenes.loc[
                        self.topologia_trenes["DPAYPOINT"] == origen_ap, 'IDFESTACION'
                    ].iloc[0]
                destino_stop = self.topologia_trenes.loc[
                        self.topologia_trenes["DPAYPOINT"] == destino_ap, 'IDFESTACION'
                    ].iloc[0]

                origen_stop,destino_stop = str(origen_stop).replace("_", "").strip(),str(destino_stop).replace("_", "").strip()

                mask = (
                    rutas_y_tiempos["ruta_paradas"].astype(str).str.contains(fr'"{origen_stop}"', regex=True)
                    & rutas_y_tiempos["ruta_paradas"].astype(str).str.contains(fr'"{destino_stop}"', regex=True)
                )

                itinerarios = rutas_y_tiempos.loc[mask].copy()

                
            for _,row in itinerarios.iterrows():
                
                ruta_paradas = row['ruta_paradas']
                tiempos = row['tiempos_entre_paradas']
                ruta_paradas = ast.literal_eval(ruta_paradas)
                tiempos = ast.literal_eval(tiempos)
    
                
                if (origen_stop in ruta_paradas) and (destino_stop in ruta_paradas):
                    i = ruta_paradas.index(origen_stop)
                    j = len(ruta_paradas) - 1 - ruta_paradas[::-1].index(destino_stop)  

                    if i < j:

                        segs = sum(self._hms_to_seconds(t) for t in tiempos[i+1 : j+1])
                        if devolver_ruta:
                            
                            return segs,ruta_paradas
                        else:
                            return segs
            ipdb.set_trace()
            raise ValueError(
            f"No encontré itinerario con origen_stop={origen_stop} antes de destino_stop={destino_stop} "
            f"(origen_ap={origen_ap}, destino_ap={destino_ap})"
            f"(tipo origen_stop={type(origen_stop)}, destino_stop={type(destino_stop)})"
            f"t ruta paradas {ruta_paradas} y orgien y destino lineas {type(ruta_paradas)}"

            
)
                

            
                
            

            
            

    def is_transbordo(self,lote_primera_sub, lote_segunda_sub,prediccion):
        """Esta regla representa el tiempo máximo de transbordo que puede
        existir entre el tiempo de bajada del segmento de viaje actual y el de subida del siguiente.
        El parámetro que la caracteriza es referido como MTT (maximum transfer time), siendo
        establecido en 18 minutos. La determinación de este valor concreto proviene de la suma de
        2 factores distintos. Por un lado, se establece un tiempo medio de espera de 10 minutos,
        según lo dispuesto en las estadísticas aportadas por Moovit. Por otro lado, se realiza el
        cálculo del tiempo que una persona tarda en recorrer los 650 metros que, de media, está
        dispuesto a andar para hacer transbordo entre estaciones. Así, considerando una velocidad
        media andando de 1.4 m/s [51], se obtiene un total de, aproximadamente, 8 minutos de
        tiempo a pie. Finalmente, la suma de ambos factores da como resultado los 18 minutos
        establecidos."""

        import pandas as pd

        buffer = pd.read_csv(BASE_DIR / "buffer_check_transbordos.csv")
        buffer_map = dict(zip(buffer["id_par"], buffer["tipo"]))
        buffer_dirty = False

        lote_primera_sub = lote_primera_sub.copy()
        lote_segunda_sub = lote_segunda_sub.copy()

        print(lote_segunda_sub['timestamp_validacion'].dtype)
        diff_sec = (
            lote_segunda_sub["timestamp_validacion"].reset_index(drop=True) - lote_primera_sub["timestamp_validacion"].reset_index(drop=True)
        ).dt.total_seconds()
                
        lote_primera_sub["prediccion_descenso"] = prediccion.to_numpy()

        from tqdm.auto import tqdm

        for (idx1, sub1), pred, diff in tqdm(
            zip(
                lote_primera_sub.iterrows(),
                prediccion.to_numpy(),
                diff_sec.to_numpy()
            ),
            total=len(lote_primera_sub),
            desc="comprobando transbordos",
        ):

            if pred == 'unknown':
                 
                lote_primera_sub.loc[idx1,'tipo'] = 'UNLINKED'
                continue
                

            origen = sub1["punto_acceso"]

            id_par = f"{origen}_y_{pred}"
            tiempo_cogido = buffer_map.get(id_par)

            if tiempo_cogido is not None and tiempo_cogido == tiempo_cogido:
                tiempo_viaje = tiempo_cogido
            else:
                tiempo_viaje = self.calculo_tiempo_estimado(origen,pred)
                buffer_map[id_par] = tiempo_viaje
                buffer_dirty = True

            tiempo_mtt = 1080

            
            if (diff <= tiempo_viaje + tiempo_mtt) and (origen != pred) and (diff >= 0): 
                lote_primera_sub.loc[idx1,'tipo'] = 'TRANSBORDO'
            else:
                lote_primera_sub.loc[idx1,'tipo'] = 'BAJADA'
                 
                
            

        if buffer_dirty:

            buffer_out = pd.DataFrame(
                {"id_par": list(buffer_map.keys()), "tipo": list(buffer_map.values())}
            )
            buffer_out.to_csv(BASE_DIR / "buffer_check_transbordos.csv", index=False)

        return lote_primera_sub
