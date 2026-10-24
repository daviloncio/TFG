"""
individual_features.py
======================
Extrae las 5 Individual Features del paper MDLF (Zhao et al., 2022)
para viajes UNLINKED de cercanías.

Idea central
------------
Para cada viaje UNLINKED (trip tr) de un usuario u, se calculan 8 sub-vectores
de dimensión N (número de estaciones cercanías posibles como destino):

  TD   N×2 : Time-Dependent      — frecuencia de destinos en mismo slot/daytype
  SD   N×1 : Space-Dependent     — frecuencia de destinos desde el mismo origen
  TSD  N×2 : Time-Space-Dep.     — SD filtrado además por slot/daytype
  TSI  N×1 : Time-Series-Indep.  — frecuencia total de destinos (sin filtro)
  PROX N×2 : Proximity           — validaciones en otros modos (bus/metro/EMT)
                                    cercanas a cada estación cercanías

La matriz N×8 de cada viaje se normaliza por fila (per-station) antes de
aplanarse al vector de salida.

Esquema de entrenamiento
------------------------
Las features se calculan sobre un historial de 1-2 meses de validaciones ya
procesadas por el pipeline de cubo_izquierdo. Solo se predicen los viajes
clasificados como UNLINKED por check_transbordo que además tengan descenso
real en crear_mapeos (para tener etiqueta supervisada).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Mapeo slot (15 min) → franja funcional. Idéntico al de region_function_lda.py.
_SLOT_TO_PERIOD: list[str] = (
    ["noche"]     * 24   # slots  0-23  → 00:00-05:45
    + ["madrug"]  *  8   # slots 24-31  → 06:00-07:45
    + ["punta_man"] * 8  # slots 32-39  → 08:00-09:45
    + ["mediodia"] * 16  # slots 40-55  → 10:00-13:45
    + ["tarde"]   * 12   # slots 56-67  → 14:00-16:45
    + ["punta_tar"] * 12 # slots 68-79  → 17:00-19:45
    + ["noche_ini"] * 16 # slots 80-95  → 20:00-23:45
)

# Columnas requeridas en history_df
_REQUIRED_HIST = {
    "id_tarjeta",
    "punto_acceso",
    "prediccion_descenso",
    "timestamp_validacion",
    "tipo_transporte",
}

# Columnas requeridas en unlinked_df
_REQUIRED_UNLINKED = {
    "id_tarjeta",
    "punto_acceso",
    "timestamp_validacion",
}

# Tipo de transporte de cercanías en el pipeline
_TIPO_CERCANIAS = "cercanias"


class IndividualFeatureExtractor:
    """
    Extrae F_u ∈ R^(N×8) normalizado por estación por viaje UNLINKED de cercanías.

    El output por viaje es un vector aplanado de N*8 valores + 1 flag (has_history),
    donde N es el número de estaciones posibles como destino.

    Parameters
    ----------
    red : str
        Prefijo del modo de transporte a predecir. '04' para cercanías.
    station_index : list[str]
        Lista ordenada de N estaciones posibles como destino (index de
        theta_cercanias.parquet).
    include_prox : bool
        Si True (MDLF gen. 2), incluye la 5ª feature PROX (N×2) → salida N×8.
        Si False (default, MDLF gen. 1 / paper base), solo 4 features → salida N×6.
        El código de PROX se conserva íntegro; este flag simplemente lo omite.
    proximity_threshold_m : int
        Radio máximo en metros para la feature PROX. Default 500.
        Debe ser ≤ 1500 (límite del CSV de paradas vecinas).
    paradas_vecinas_path : str | Path | None
        Ruta al fichero topologia/paradas_vecinas_1500m.csv.
        Si es None, la feature PROX devuelve ceros y se emite un aviso.
    norm : str or None
        Tipo de normalización per-estación tras apilar las sub-features.
        'l2' (default), 'l1' o 'max'. Ver sklearn.preprocessing.normalize.
        None → sin normalización (frecuencias crudas, igual que el paper).
    slot_minutes : int
        Duración del slot temporal en minutos. Default 15.
    td_mode : str
        Controla cómo se construyen los 2 sub-vectores de TD y TSD:

        'context' (default) — fiel al paper MDLF (Zhao et al., 2022), ecs. (2)(3)(5)(6):
            TD  = [mismo_daytype+franja,  cualquier_daytype+franja]
            TSD = [mismo_daytype+franja+origen, cualquier_daytype+franja+origen]
            El segundo vector incluye todos los días (suavizado natural por escasez).

        'split' — alternativa no presente en el paper:
            TD  = [conteo_WD+franja, conteo_WE+franja]
            TSD = [conteo_WD+franja+origen, conteo_WE+franja+origen]
            El modelo ve ambos tipos de día simultáneamente, sin condicionar
            al daytype del viaje actual.

        Dimensión de salida igual en ambos casos (N×6 sin PROX, N×8 con PROX).
    """

    def __init__(
        self,
        red: str,
        station_index: list[str],
        include_prox: bool = False,
        proximity_threshold_m: int = 500,
        paradas_vecinas_path: str | Path | None = None,
        norm: str | None = "l2",
        slot_minutes: int = 15,
        td_mode: str = "context",
        holiday_dates: frozenset = frozenset(),
    ) -> None:
        if td_mode not in ("context", "split"):
            raise ValueError(f"td_mode debe ser 'context' o 'split', recibido: {td_mode!r}")
        self.red = red
        self.station_index = list(station_index)
        self.N = len(self.station_index)
        self._si_pos: dict[str, int] = {s: i for i, s in enumerate(self.station_index)}
        self.include_prox = include_prox
        self.proximity_threshold_m = proximity_threshold_m
        self.paradas_vecinas_path = Path(paradas_vecinas_path) if paradas_vecinas_path else None
        self.norm = norm
        self.slot_minutes = slot_minutes
        self.td_mode = td_mode
        self.holiday_dates = frozenset(holiday_dates)
        # Dimensión por estación: 6 sin PROX (paper gen. 1), 8 con PROX (gen. 2)
        self.n_sub_features = 8 if include_prox else 6

        # Estado interno — solo existe tras fit()
        self._hist_red: dict[str, pd.DataFrame] | None = None
        self._hist_other: dict[str, pd.DataFrame] | None = None
        self._prox_lookup: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, history_df: pd.DataFrame) -> "IndividualFeatureExtractor":
        """
        Indexa el historial de usuarios y construye el lookup de proximidad.

        Parameters
        ----------
        history_df : pd.DataFrame
            Historial completo (TODOS los modos, 1-2 meses procesados).
            Columnas requeridas:
                id_tarjeta, punto_acceso, prediccion_descenso,
                timestamp_validacion, tipo_transporte

        Returns
        -------
        self
        """
        _validate_columns(history_df, _REQUIRED_HIST)

        h = _add_temporal(history_df, self.slot_minutes, self.holiday_dates)

        # Separar historial cercanías (features 1-4) y otros modos (feature PROX)
        mask_red = h["tipo_transporte"] == _TIPO_CERCANIAS
        h_red   = h[mask_red].rename(
            columns={"punto_acceso": "origen", "prediccion_descenso": "destino"}
        ).copy()
        h_other = h[~mask_red].rename(
            columns={"punto_acceso": "origen"}
        ).copy()

        # Índice por usuario
        self._hist_red   = {uid: g.reset_index(drop=True)
                            for uid, g in h_red.groupby("id_tarjeta")}
        self._hist_other = {uid: g.reset_index(drop=True)
                            for uid, g in h_other.groupby("id_tarjeta")}

        # Lookup de proximidad (cargado una sola vez)
        self._prox_lookup = self._build_prox_lookup()

        logger.info(
            "fit: %d usuarios con historial %s | %d con historial otros modos",
            len(self._hist_red), self.red, len(self._hist_other),
        )
        return self

    def transform(self, unlinked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula F_u ∈ R^(N×8) normalizado para cada viaje UNLINKED.

        Parameters
        ----------
        unlinked_df : pd.DataFrame
            Viajes UNLINKED de cercanías a predecir.
            Columnas requeridas: id_tarjeta, punto_acceso, timestamp_validacion.

        Returns
        -------
        pd.DataFrame
            Shape (n_unlinked, N*8 + 1).
            Columnas: {si}__{feat} para cada si ∈ station_index y
                      feat ∈ {td_a, td_ap, sd, tsd_b, tsd_bp, tsi, prox_tsi, prox_td}
                    + has_history (0/1).
            Index = RangeIndex del unlinked_df original.
        """
        if self._hist_red is None:
            raise RuntimeError("Llama a fit() antes de transform().")

        _validate_columns(unlinked_df, _REQUIRED_UNLINKED)

        unlinked = _add_temporal(
            unlinked_df.reset_index(drop=True), self.slot_minutes, self.holiday_dates
        ).rename(columns={"punto_acceso": "origen"})

        # Construir nombres de columnas de salida
        if self.td_mode == "split":
            td_feats = ("td_wd", "td_we", "sd", "tsd_wd", "tsd_we", "tsi")
        else:
            td_feats = ("td_a", "td_ap", "sd", "tsd_b", "tsd_bp", "tsi")
        if self.include_prox:
            sub_feats = td_feats + ("prox_tsi", "prox_td")
        else:
            sub_feats = td_feats
        col_names = [f"{si}__{feat}" for si in self.station_index for feat in sub_feats]
        col_names.append("has_history")

        rows = np.empty((len(unlinked), self.N * self.n_sub_features + 1), dtype=np.float32)

        for idx, row in unlinked.iterrows():
            uid     = row["id_tarjeta"]
            s_o     = row["origen"]
            franja  = row["franja"]
            daytype = row["daytype"]
            ts      = row["timestamp_validacion"]

            # Solo historial ANTERIOR al viaje objetivo (evitar fuga de datos)
            hist_u = self._hist_red.get(uid)
            if hist_u is not None:
                hist_u = hist_u[hist_u["timestamp_validacion"] < ts]
                if len(hist_u) == 0:
                    hist_u = None

            other_u = self._hist_other.get(uid)
            if other_u is not None:
                other_u = other_u[other_u["timestamp_validacion"] < ts]
                if len(other_u) == 0:
                    other_u = None

            has_history = 1 if hist_u is not None else 0

            vec = self._compute_features(hist_u, other_u, s_o, franja, daytype)
            rows[idx, :-1] = vec
            rows[idx, -1]  = has_history

        result = pd.DataFrame(rows, columns=col_names)
        logger.info("transform: %d viajes UNLINKED procesados", len(result))
        return result

    def fit_transform(
        self,
        history_df: pd.DataFrame,
        unlinked_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ajusta y transforma en un único paso."""
        return self.fit(history_df).transform(unlinked_df)

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        hist_u: pd.DataFrame | None,
        other_u: pd.DataFrame | None,
        s_o: str,
        franja: str,
        daytype: str,
    ) -> np.ndarray:
        """
        Calcula los 8 sub-vectores de N dimensiones y los normaliza por estación.

        Retorna un array 1D de N*8 valores (filas apiladas de la matriz N×8).
        """
        # --- features 1-4: historial cercanías ---
        if hist_u is not None and len(hist_u) > 0:
            mask_slot    = hist_u["franja"]  == franja
            mask_day     = hist_u["daytype"] == daytype
            mask_orig    = hist_u["origen"]  == s_o

            tsi = _count_dest(hist_u,           self._si_pos, self.N)
            sd  = _count_dest(hist_u[mask_orig], self._si_pos, self.N)

            if self.td_mode == "split":
                # Paper: columna WD y columna WE por separado
                mask_wd = hist_u["daytype"] == "WD"
                mask_we = hist_u["daytype"] == "WE"
                td_1  = _count_dest(hist_u[mask_slot & mask_wd],        self._si_pos, self.N)
                td_2  = _count_dest(hist_u[mask_slot & mask_we],        self._si_pos, self.N)
                tsd_1 = _count_dest(hist_u[mask_orig & mask_slot & mask_wd], self._si_pos, self.N)
                tsd_2 = _count_dest(hist_u[mask_orig & mask_slot & mask_we], self._si_pos, self.N)
            else:
                # context: mismo daytype del viaje + versión relajada (cualquier daytype)
                td_1  = _count_dest(hist_u[mask_slot & mask_day],       self._si_pos, self.N)
                td_2  = _count_dest(hist_u[mask_slot],                   self._si_pos, self.N)
                tsd_1 = _count_dest(hist_u[mask_orig & mask_slot & mask_day], self._si_pos, self.N)
                tsd_2 = _count_dest(hist_u[mask_orig & mask_slot],       self._si_pos, self.N)
        else:
            tsi = sd = td_1 = td_2 = tsd_1 = tsd_2 = np.zeros(self.N, dtype=np.float32)

        # --- feature 5: proximidad en otros modos (solo gen. 2, include_prox=True) ---
        if self.include_prox:
            prox_tsi, prox_td = self._compute_prox(other_u, franja, daytype)
            vecs = [td_1, td_2, sd, tsd_1, tsd_2, tsi, prox_tsi, prox_td]
        else:
            vecs = [td_1, td_2, sd, tsd_1, tsd_2, tsi]

        # Apilar → (N, n_sub_features), normalizar por fila (opcional), aplanar
        mat = np.stack(vecs, axis=1).astype(np.float32)
        if self.norm:
            mat = normalize(mat, norm=self.norm, axis=1)
        return mat.flatten()

    def _compute_prox(
        self,
        other_u: pd.DataFrame | None,
        franja: str,
        daytype: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula los vectores de proximidad (prox_tsi, prox_td).

        Para cada validación histórica del usuario en otros modos (bus, metro, EMT...),
        busca las estaciones cercanías dentro del radio y les suma +1.
        """
        prox_tsi = np.zeros(self.N, dtype=np.float32)
        prox_td  = np.zeros(self.N, dtype=np.float32)

        if (other_u is None or other_u.empty or not self._prox_lookup):
            return prox_tsi, prox_td

        # Añadir lista de cercanías cercanas para cada validación
        other_u = other_u.copy()
        other_u["cercanias_cercanas"] = other_u["origen"].map(
            lambda p: self._prox_lookup.get(p, [])
        )
        # Solo filas con al menos una estación cercana
        has_prox = other_u["cercanias_cercanas"].map(len) > 0
        if not has_prox.any():
            return prox_tsi, prox_td

        exploded = other_u[has_prox].explode("cercanias_cercanas")
        col = "cercanias_cercanas"

        # prox_tsi: todas las validaciones históricas
        for si, cnt in exploded[col].value_counts().items():
            pos = self._si_pos.get(si)
            if pos is not None:
                prox_tsi[pos] = cnt

        # prox_td: filtrado por misma franja + mismo daytype
        mask = (exploded["franja"] == franja) & (exploded["daytype"] == daytype)
        for si, cnt in exploded.loc[mask, col].value_counts().items():
            pos = self._si_pos.get(si)
            if pos is not None:
                prox_td[pos] = cnt

        return prox_tsi, prox_td

    def _build_prox_lookup(self) -> dict[str, list[str]]:
        """
        Carga paradas_vecinas_1500m.csv, filtra por cercanías y radio,
        y devuelve un dict {punto_acceso_otro_modo → [cercanias_stations_cercanas]}.
        """
        if self.paradas_vecinas_path is None:
            logger.warning(
                "paradas_vecinas_path no especificado → feature PROX será cero para todos."
            )
            return {}
        if not self.paradas_vecinas_path.exists():
            raise FileNotFoundError(
                f"No se encuentra el fichero de paradas vecinas: {self.paradas_vecinas_path}"
            )

        logger.info("Cargando paradas_vecinas desde %s ...", self.paradas_vecinas_path)
        vecinas = pd.read_csv(
            self.paradas_vecinas_path,
            usecols=["punto_acceso", "vecina", "tipo_transporte_vecina", "distancia_m"],
        )
        mask = (
            (vecinas["tipo_transporte_vecina"] == "cercanias") &
            (vecinas["distancia_m"] <= self.proximity_threshold_m)
        )
        vecinas_c = vecinas[mask]
        lookup = vecinas_c.groupby("punto_acceso")["vecina"].apply(list).to_dict()
        logger.info(
            "Prox lookup: %d paradas con vecina cercanías a ≤%dm (de %d filas originales)",
            len(lookup), self.proximity_threshold_m, len(vecinas),
        )
        return lookup

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = "fitted" if self._hist_red is not None else "not fitted"
        return (
            f"IndividualFeatureExtractor("
            f"red={self.red!r}, N={self.N}, "
            f"td_mode={self.td_mode!r}, "
            f"proximity_threshold_m={self.proximity_threshold_m}, "
            f"norm={self.norm!r}, status={fitted})"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _add_temporal(
    df: pd.DataFrame,
    slot_minutes: int = 15,
    holiday_dates: frozenset = frozenset(),
) -> pd.DataFrame:
    """Añade columnas 'franja' y 'daytype' desde 'timestamp_validacion'.

    holiday_dates : fechas adicionales (festivos laborables) que se tratan
    como 'WE'. Formato: frozenset de datetime.date.
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp_validacion"], utc=False).dt.tz_localize(None)
    df["timestamp_validacion"] = ts  # normaliza a naive para comparaciones internas
    slot = (ts.dt.hour * 60 + ts.dt.minute) // slot_minutes
    df["franja"]  = slot.map(lambda s: _SLOT_TO_PERIOD[s])
    is_we = (ts.dt.dayofweek >= 5) | ts.dt.date.map(lambda d: d in holiday_dates)
    df["daytype"] = np.where(is_we, "WE", "WD")
    return df


def _count_dest(
    df: pd.DataFrame,
    si_pos: dict[str, int],
    N: int,
) -> np.ndarray:
    """
    Cuenta ocurrencias de 'destino' en df y las mapea a un vector de N dimensiones.
    Destinos fuera de si_pos se descartan silenciosamente.
    """
    vec = np.zeros(N, dtype=np.float32)
    if df.empty or "destino" not in df.columns:
        return vec
    for dest, cnt in df["destino"].value_counts().items():
        pos = si_pos.get(dest)
        if pos is not None:
            vec[pos] = cnt
    return vec


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing}")
