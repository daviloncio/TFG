"""
neighborhood_crowd.py
=====================
Crowd features filtradas al "vecindario de residencia" del viajero.

Idea central
------------
El ``CrowdFeatureExtractor`` estándar usa TODOS los usuarios que parten
desde la misma estación origen. Este extractor, en cambio, usa solo
los usuarios que comparten la **parada de residencia** con el viajero,
bajo el supuesto de que personas del mismo barrio tienen patrones de
movilidad similares.

Flujo
-----
1. ``ResidenceInferrer.fit()`` ya ha determinado la parada de residencia
   de cada usuario en el historial.
2. ``NeighborhoodCrowdExtractor.fit()`` agrega el historial de cercanías
   por ``(residence_station, daytype, slot, fecha, destino)`` en lugar de
   ``(origen, daytype, slot, fecha, destino)``.
3. ``transform()`` consulta ese índice para cada viaje usando la residencia
   del usuario. Si el usuario no tiene residencia confirmada, o su
   vecindario tiene menos de ``min_neighbors`` usuarios distintos, devuelve
   un vector de ceros y ``has_nbcrowd_signal = 0``.

min_neighbors
-------------
Protege contra vecindarios con muy pocos usuarios (2-3 personas), donde
la estimación de probabilidad de destino sería extremadamente ruidosa.
Con ``min_neighbors=10`` (default), solo se usa el vecindario si al
menos 10 usuarios distintos comparten esa parada de residencia en el
historial de cercanías.

Zeros + flag
------------
Para usuarios sin señal de vecindario el vector es todo ceros.
La CNN aprende a ignorar esas dimensiones: los gradientes que llegan a
los pesos del Conv1d para esas posiciones son proporcionales a la entrada
(que es 0), así que los pesos asociados a la feature de vecindario no se
actualizan en los viajes sin señal y aprenden únicamente de los viajes
con señal. El flag ``has_nbcrowd_signal`` permite auditar la cobertura.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from residence_inferrer import ResidenceInferrer

logger = logging.getLogger(__name__)

# Columnas requeridas
_REQUIRED_HIST = {
    "id_tarjeta",
    "punto_acceso",
    "prediccion_descenso",
    "timestamp_validacion",
    "tipo_transporte",
}
_REQUIRED_UNLINKED = {
    "id_tarjeta",
    "punto_acceso",
    "timestamp_validacion",
}

_TIPO_CERCANIAS = "cercanias"

# Mapeo slot (15 min) → franja funcional. Idéntico a crowd_features.py.
_SLOT_TO_PERIOD: list[str] = (
    ["noche"]      * 24
    + ["madrug"]   *  8
    + ["punta_man"] * 8
    + ["mediodia"] * 16
    + ["tarde"]    * 12
    + ["punta_tar"] * 12
    + ["noche_ini"] * 16
)


class NeighborhoodCrowdExtractor:
    """
    Crowd features del vecindario de residencia por viaje UNLINKED de cercanías.

    El output por viaje es un vector aplanado de N*n_weeks valores + 1 flag
    (has_nbcrowd_signal), idéntico en forma al de ``CrowdFeatureExtractor``.

    Parameters
    ----------
    red : str
        Prefijo del modo a predecir. '04' para cercanías.
    station_index : list[str]
        Lista ordenada de N estaciones destino posibles.
    n_weeks : int
        Semanas de historial colectivo hacia atrás. Default 2.
    min_neighbors : int
        Mínimo de usuarios distintos en el vecindario para usar la señal.
        Vecindarios más pequeños devuelven ceros. Default 10.
    norm : str or None
        Normalización per-station: 'l2' (default), 'l1' o 'max'.
        None → sin normalización (proporciones crudas, igual que el paper).
    slot_minutes : int
        Duración del slot en minutos. Default 15 (96 slots/día).
    """

    def __init__(
        self,
        red: str,
        station_index: list[str],
        n_weeks: int = 2,
        min_neighbors: int = 10,
        norm: str | None = "l2",
        slot_minutes: int = 15,
        holiday_dates: frozenset = frozenset(),
    ) -> None:
        self.red = red
        self.station_index = list(station_index)
        self.N = len(self.station_index)
        self._si_pos: dict[str, int] = {s: i for i, s in enumerate(self.station_index)}
        self.n_weeks = n_weeks
        self.min_neighbors = min_neighbors
        self.norm = norm
        self.slot_minutes = slot_minutes
        self.holiday_dates = frozenset(holiday_dates)

        # Estado interno — solo existe tras fit()
        # dict[(residence_station, daytype, slot)] → DataFrame(fecha, destino, n)
        self._agg_by_residence: dict[tuple, pd.DataFrame] | None = None
        # dict[residence_station] → n_usuarios_distintos
        self._neighbors_count: dict[str, int] | None = None
        # copia del mapa user→residence del inferrer
        self._residence: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        history_df: pd.DataFrame,
        inferrer: ResidenceInferrer,
    ) -> "NeighborhoodCrowdExtractor":
        """
        Agrega el historial de cercanías por vecindario de residencia.

        Parameters
        ----------
        history_df : pd.DataFrame
            Historial completo (todos los modos).
            Columnas: id_tarjeta, punto_acceso, prediccion_descenso,
                      timestamp_validacion, tipo_transporte.
        inferrer : ResidenceInferrer
            Ya ajustado (fit() ya llamado).

        Returns
        -------
        self
        """
        _validate_columns(history_df, _REQUIRED_HIST)

        # Guardar mapa user→residence
        self._residence = dict(inferrer._residence)  # copia defensiva

        # Solo cercanías
        h = _add_temporal(
            history_df[history_df["tipo_transporte"] == _TIPO_CERCANIAS].copy(),
            self.slot_minutes,
            self.holiday_dates,
        )
        h = h.rename(
            columns={
                "punto_acceso": "origen",
                "prediccion_descenso": "destino",
            }
        )
        h["fecha"] = pd.to_datetime(h["timestamp_validacion"]).dt.tz_localize(None).dt.date
        h = h[h["destino"].notna() & (h["destino"] != "unknown")]

        # Añadir columna residence_station por usuario
        h["residence_station"] = h["id_tarjeta"].map(self._residence)
        h = h[h["residence_station"].notna()]   # descartar usuarios sin residencia

        if h.empty:
            logger.warning(
                "NeighborhoodCrowdExtractor.fit(): ningún usuario del historial "
                "tiene residencia confirmada. El extractor devolverá siempre ceros."
            )
            self._agg_by_residence = {}
            self._neighbors_count = {}
            return self

        # Contar usuarios distintos por residence_station
        self._neighbors_count = (
            h.groupby("residence_station")["id_tarjeta"]
            .nunique()
            .to_dict()
        )

        # Filtrar vecindarios con menos de min_neighbors usuarios
        valid_residences = {
            r for r, n in self._neighbors_count.items()
            if n >= self.min_neighbors
        }
        h = h[h["residence_station"].isin(valid_residences)]

        n_filtered = len(self._neighbors_count) - len(valid_residences)
        if n_filtered > 0:
            logger.info(
                "%d paradas de residencia descartadas por tener < %d vecinos",
                n_filtered, self.min_neighbors,
            )

        if h.empty:
            logger.warning(
                "Tras aplicar min_neighbors=%d, no quedan vecindarios válidos.",
                self.min_neighbors,
            )
            self._agg_by_residence = {}
            return self

        # Agregar por (residence_station, daytype, slot, fecha, destino)
        agg = (
            h.groupby(["residence_station", "daytype", "slot", "fecha", "destino"])
            .size()
            .reset_index(name="n")
        )

        self._agg_by_residence = {
            key: grp[["fecha", "destino", "n"]]
            .sort_values("fecha")
            .reset_index(drop=True)
            for key, grp in agg.groupby(["residence_station", "daytype", "slot"])
        }

        logger.info(
            "fit: %d vecindarios válidos (≥%d vecinos) | "
            "%d claves (residence, daytype, slot) | "
            "%d registros agregados",
            len(valid_residences), self.min_neighbors,
            len(self._agg_by_residence), len(agg),
        )
        return self

    def transform(
        self,
        unlinked_df: pd.DataFrame,
        reference_date: "date | None" = None,
    ) -> pd.DataFrame:
        """
        Calcula F_nc ∈ R^(N × n_weeks) normalizado para cada viaje UNLINKED.

        Parameters
        ----------
        unlinked_df : pd.DataFrame
            Viajes UNLINKED de cercanías a predecir.
            Columnas requeridas: id_tarjeta, punto_acceso, timestamp_validacion.
        reference_date : date | None
            Si se especifica, la fecha de cada viaje se capa a
            min(fecha_viaje, reference_date) antes de calcular las ventanas
            de lookback. Mismo propósito que en CrowdFeatureExtractor.transform().

        Returns
        -------
        pd.DataFrame
            Shape (n_unlinked, N*n_weeks + 1).
            Columnas: {si}__nbw{m} + has_nbcrowd_signal.
            Ceros para viajes sin señal de vecindario.
        """
        if self._agg_by_residence is None:
            raise RuntimeError("Llama a fit() antes de transform().")

        _validate_columns(unlinked_df, _REQUIRED_UNLINKED)

        unlinked = _add_temporal(
            unlinked_df.reset_index(drop=True).copy(), self.slot_minutes, self.holiday_dates
        ).rename(columns={"punto_acceso": "origen"})
        unlinked["fecha"] = pd.to_datetime(
            unlinked["timestamp_validacion"]
        ).dt.tz_localize(None).dt.date
        if reference_date is not None:
            unlinked["fecha"] = unlinked["fecha"].apply(
                lambda d: min(d, reference_date)
            )

        sub_feats = [f"nbw{m}" for m in range(1, self.n_weeks + 1)]
        col_names = [f"{si}__{feat}" for si in self.station_index for feat in sub_feats]
        col_names.append("has_nbcrowd_signal")

        rows = np.empty((len(unlinked), self.N * self.n_weeks + 1), dtype=np.float32)

        for idx, row in unlinked.iterrows():
            uid     = row["id_tarjeta"]
            daytype = row["daytype"]
            slot    = int(row["slot"])
            fecha   = row["fecha"]

            residence = self._residence.get(uid) if self._residence else None

            # Sin residencia o vecindario insuficiente → ceros
            if (
                residence is None
                or self._neighbors_count.get(residence, 0) < self.min_neighbors
            ):
                rows[idx, :-1] = 0.0
                rows[idx, -1]  = 0
                continue

            vec, has_signal = self._compute_features(residence, daytype, slot, fecha)
            rows[idx, :-1] = vec
            rows[idx, -1]  = has_signal

        result = pd.DataFrame(rows, columns=col_names)
        logger.info("transform: %d viajes UNLINKED procesados", len(result))
        return result

    def fit_transform(
        self,
        history_df: pd.DataFrame,
        inferrer: ResidenceInferrer,
        unlinked_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ajusta y transforma en un único paso."""
        return self.fit(history_df, inferrer).transform(unlinked_df)

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        residence: str,
        daytype: str,
        slot: int,
        trip_date: date,
    ) -> tuple[np.ndarray, int]:
        """Calcula n_weeks vectores de N dims y los normaliza per-station."""
        key = (residence, daytype, slot)
        week_vecs = []
        any_signal = False

        for m in range(1, self.n_weeks + 1):
            t_end   = trip_date - timedelta(days=(m - 1) * 7)
            t_start = trip_date - timedelta(days=m * 7)
            vec = self._compute_crowd_week(key, t_start, t_end)
            if vec.any():
                any_signal = True
            week_vecs.append(vec)

        mat = np.stack(week_vecs, axis=1).astype(np.float32)   # (N, n_weeks)
        if self.norm:
            mat = normalize(mat, norm=self.norm, axis=1)
        return mat.flatten(), int(any_signal)

    def _compute_crowd_week(
        self,
        key: tuple,
        t_start: date,
        t_end: date,
    ) -> np.ndarray:
        """Proporciones de destinos para (residence, daytype, slot) en [t_start, t_end)."""
        vec = np.zeros(self.N, dtype=np.float32)

        grp = self._agg_by_residence.get(key)
        if grp is None or grp.empty:
            return vec

        mask = (grp["fecha"] >= t_start) & (grp["fecha"] < t_end)
        subset = grp.loc[mask, ["destino", "n"]]
        if subset.empty:
            return vec

        dest_counts = subset.groupby("destino")["n"].sum()
        total = dest_counts.sum()
        if total == 0:
            return vec

        for dest, cnt in dest_counts.items():
            pos = self._si_pos.get(dest)
            if pos is not None:
                vec[pos] = cnt / total

        return vec

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = "fitted" if self._agg_by_residence is not None else "not fitted"
        n_nb = len(self._neighbors_count) if self._neighbors_count else 0
        return (
            f"NeighborhoodCrowdExtractor("
            f"red={self.red!r}, N={self.N}, "
            f"n_weeks={self.n_weeks}, "
            f"min_neighbors={self.min_neighbors}, "
            f"n_residences={n_nb}, status={fitted})"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _add_temporal(
    df: pd.DataFrame,
    slot_minutes: int = 15,
    holiday_dates: frozenset = frozenset(),
) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp_validacion"], utc=False).dt.tz_localize(None)
    df["timestamp_validacion"] = ts  # normaliza a naive para comparaciones internas
    df["slot"]    = (ts.dt.hour * 60 + ts.dt.minute) // slot_minutes
    is_we = (ts.dt.dayofweek >= 5) | ts.dt.date.map(lambda d: d in holiday_dates)
    df["daytype"] = np.where(is_we, "WE", "WD")
    return df


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing}")
