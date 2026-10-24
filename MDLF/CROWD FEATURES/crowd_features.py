"""
crowd_features.py
=================
Extrae las Crowd Features del paper MDLF (Zhao et al., 2022), Sección C,
para viajes UNLINKED de cercanías.

Idea central
------------
El historial individual del usuario puede ser escaso. Las crowd features
complementan ese historial usando el comportamiento colectivo de TODOS los
usuarios desde la misma estación origen s_o, en la misma franja horaria,
mirando hacia atrás semana a semana (periodicidad de 7 días).

Para cada viaje UNLINKED (trip tr) con origen s_o, daytype d, franja f^l, timestamp t:
  P(destino | origen=s_o, daytype=d, franja=f^l) estimada sobre el colectivo.

  - Para cada semana m = 1, ..., n_weeks:
      * Filtrar TODOS los viajes históricos de cercanías donde:
          origen == s_o  AND  daytype == d  AND  franja == f^l  AND
          fecha ∈ [t − m×7 días, t − (m−1)×7 días)
      * Contar destinos → proporciones → vector F_c^m ∈ R^N
  - Apilar: F_c = [F_c^1 || ... || F_c^n_weeks] → matriz N × n_weeks

Se filtra por daytype (WD / WE) para separar patrones laborables de fin de
semana/festivo en la misma franja horaria.

La matriz N×n_weeks se normaliza por fila (per-station) igual que
IndividualFeatureExtractor con su N×8.

Esquema de entrenamiento
------------------------
fit() recibe el mismo history_df que IndividualFeatureExtractor (todos los
modos). Internamente filtra cercanías y pre-agrega por (origen, franja, fecha)
para búsqueda eficiente en transform().
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Mapeo slot (15 min) → franja funcional. Idéntico al de individual_features.py.
_SLOT_TO_PERIOD: list[str] = (
    ["noche"]      * 24   # slots  0-23  → 00:00-05:45
    + ["madrug"]   *  8   # slots 24-31  → 06:00-07:45
    + ["punta_man"] * 8   # slots 32-39  → 08:00-09:45
    + ["mediodia"] * 16   # slots 40-55  → 10:00-13:45
    + ["tarde"]    * 12   # slots 56-67  → 14:00-16:45
    + ["punta_tar"] * 12  # slots 68-79  → 17:00-19:45
    + ["noche_ini"] * 16  # slots 80-95  → 20:00-23:45
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

_TIPO_CERCANIAS = "cercanias"


class CrowdFeatureExtractor:
    """
    Extrae F_c ∈ R^(N × n_weeks) normalizado por estación, por viaje UNLINKED.

    El output por viaje es un vector aplanado de N*n_weeks valores + 1 flag
    (has_crowd_signal), donde N es el número de estaciones posibles como destino
    y n_weeks el número de semanas de historial colectivo.

    Parameters
    ----------
    red : str
        Prefijo del modo de transporte a predecir. '04' para cercanías.
    station_index : list[str]
        Lista ordenada de N estaciones posibles como destino (index de
        theta_cercanias.parquet).
    n_weeks : int
        Número de semanas hacia atrás a considerar. Default 2.
        El paper original usa 4; 1 = solo la semana inmediatamente anterior.
    norm : str or None
        Tipo de normalización per-estación tras apilar las n_weeks columnas.
        'l2' (default), 'l1' o 'max'. Ver sklearn.preprocessing.normalize.
        None → sin normalización (proporciones crudas, igual que el paper).
    slot_minutes : int
        Duración del slot temporal en minutos. Default 30 → 48 slots/día.
    """

    def __init__(
        self,
        red: str,
        station_index: list[str],
        n_weeks: int = 2,
        norm: str | None = "l2",
        slot_minutes: int = 15,
        holiday_dates: frozenset = frozenset(),
    ) -> None:
        self.red = red
        self.station_index = list(station_index)
        self.N = len(self.station_index)
        self._si_pos: dict[str, int] = {s: i for i, s in enumerate(self.station_index)}
        self.n_weeks = n_weeks
        self.norm = norm
        self.slot_minutes = slot_minutes
        self.holiday_dates = frozenset(holiday_dates)

        # Estado interno — solo existe tras fit()
        # dict[(origen, daytype, slot)] → DataFrame(fecha, destino, n) ordenado por fecha
        # Condiciona sobre P(destino | origen, contexto_temporal) donde
        # contexto_temporal = (daytype ∈ {WD, WE}, slot exacto de slot_minutes min)
        self._agg: dict[tuple[str, str, int], pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, history_df: pd.DataFrame) -> "CrowdFeatureExtractor":
        """
        Pre-agrega el historial colectivo de cercanías por (origen, franja, fecha).

        Parameters
        ----------
        history_df : pd.DataFrame
            Historial completo (TODOS los modos, misma entrada que
            IndividualFeatureExtractor).
            Columnas requeridas:
                id_tarjeta, punto_acceso, prediccion_descenso,
                timestamp_validacion, tipo_transporte

        Returns
        -------
        self
        """
        _validate_columns(history_df, _REQUIRED_HIST)

        # Solo cercanías: el OD crowd se agrega a nivel estación cercanías
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
        h["fecha"] = pd.to_datetime(h["timestamp_validacion"]).dt.date

        # Descartar filas sin destino válido
        h = h[h["destino"].notna() & (h["destino"] != "unknown")]

        # Agregar: conteo de (origen, daytype, slot, fecha, destino)
        # Condiciona en P(destino | origen, daytype, slot)
        agg = (
            h.groupby(["origen", "daytype", "slot", "fecha", "destino"])
            .size()
            .reset_index(name="n")
        )

        # Índice por (origen, daytype, slot) para búsqueda O(log n) por fecha
        self._agg = {
            key: grp[["fecha", "destino", "n"]]
            .sort_values("fecha")
            .reset_index(drop=True)
            for key, grp in agg.groupby(["origen", "daytype", "slot"])
        }

        logger.info(
            "fit: %d claves (origen, daytype, slot) en el índice crowd | "
            "%d registros agregados | cercanías en historial: %d filas",
            len(self._agg),
            len(agg),
            len(h),
        )
        return self

    def transform(
        self,
        unlinked_df: pd.DataFrame,
        reference_date: "date | None" = None,
    ) -> pd.DataFrame:
        """
        Calcula F_c ∈ R^(N × n_weeks) normalizado para cada viaje UNLINKED.

        Parameters
        ----------
        unlinked_df : pd.DataFrame
            Viajes UNLINKED de cercanías a predecir.
            Columnas requeridas: id_tarjeta, punto_acceso, timestamp_validacion.
        reference_date : date | None
            Si se especifica, la fecha de cada viaje se capa a
            min(fecha_viaje, reference_date) antes de calcular las ventanas
            de lookback. Útil en test con split temporal: ancla el lookback
            al último día de train para que todas las semanas tengan datos.
            Si None (por defecto), se usa la fecha real del viaje.

        Returns
        -------
        pd.DataFrame
            Shape (n_unlinked, N*n_weeks + 1).
            Columnas: {si}__w{m} para cada si ∈ station_index y
                      m ∈ 1..n_weeks (1 = semana más reciente)
                    + has_crowd_signal (0/1).
            Index = RangeIndex del unlinked_df original.
        """
        if self._agg is None:
            raise RuntimeError("Llama a fit() antes de transform().")

        _validate_columns(unlinked_df, _REQUIRED_UNLINKED)

        unlinked = _add_temporal(
            unlinked_df.reset_index(drop=True).copy(), self.slot_minutes, self.holiday_dates
        ).rename(columns={"punto_acceso": "origen"})
        unlinked["fecha"] = pd.to_datetime(unlinked["timestamp_validacion"]).dt.date
        if reference_date is not None:
            unlinked["fecha"] = unlinked["fecha"].apply(
                lambda d: min(d, reference_date)
            )
        unlinked["slot"]  = unlinked["slot"].astype(int)

        combo_cols = ["origen", "daytype", "slot", "fecha"]
        D = self.N * self.n_weeks

        # ── Paso 1: combos únicos (origen, daytype, slot, fecha) ─────────────
        # Muchos viajes comparten origen/slot/fecha → calculamos features
        # una sola vez por combo y luego mapeamos con indexación vectorial.
        unique_combos = (
            unlinked[combo_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        n_unique = len(unique_combos)
        logger.info(
            "transform: %d viajes → %d combos únicos (origen, daytype, slot, fecha)",
            len(unlinked), n_unique,
        )

        # ── Paso 2: features por combo único ─────────────────────────────────
        # RAM extra: n_unique × (N*n_weeks+1) × 4 bytes — mucho menor que el output
        feat_mat = np.zeros((n_unique, D + 1), dtype=np.float32)
        for i, row in unique_combos.iterrows():
            vec, has_signal = self._compute_features(
                row["origen"], row["daytype"], int(row["slot"]), row["fecha"]
            )
            feat_mat[i, :D] = vec
            feat_mat[i,  D] = has_signal

        # ── Paso 3: mapear de vuelta a cada viaje por indexación vectorial ───
        unique_combos["_combo_id"] = np.arange(n_unique)
        unlinked = unlinked.merge(unique_combos, on=combo_cols, how="left")
        combo_ids = unlinked["_combo_id"].to_numpy(dtype=int)

        rows = feat_mat[combo_ids]  # (n_unlinked, D+1) — copia vectorizada

        col_names = [
            f"{si}__w{m}"
            for si in self.station_index
            for m in range(1, self.n_weeks + 1)
        ]
        col_names.append("has_crowd_signal")

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
        s_o: str,
        daytype: str,
        slot: int,
        trip_date: date,
    ) -> tuple[np.ndarray, int]:
        """
        Calcula los n_weeks vectores de N dimensiones y los normaliza por estación.

        Filtra por (origen, daytype, slot) para obtener P(destino | origen, ctx_temporal).
        Retorna (array 1D de N*n_weeks valores, has_signal flag).
        """
        key = (s_o, daytype, slot)
        week_vecs = []
        any_signal = False

        for m in range(1, self.n_weeks + 1):
            t_end   = trip_date - timedelta(days=(m - 1) * 7)
            t_start = trip_date - timedelta(days=m * 7)
            vec = self._compute_crowd_week(key, t_start, t_end)
            if vec.any():
                any_signal = True
            week_vecs.append(vec)

        # Apilar → (N, n_weeks), normalizar por fila (opcional), aplanar
        mat = np.stack(week_vecs, axis=1).astype(np.float32)   # (N, n_weeks)
        if self.norm:
            mat = normalize(mat, norm=self.norm, axis=1)
        return mat.flatten(), int(any_signal)

    def _compute_crowd_week(
        self,
        key: tuple[str, str],
        t_start: date,
        t_end: date,
    ) -> np.ndarray:
        """
        Para la clave (origen, franja) y la ventana [t_start, t_end),
        devuelve el vector de proporciones de destinos de longitud N.
        """
        vec = np.zeros(self.N, dtype=np.float32)

        grp = self._agg.get(key)
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
        fitted = "fitted" if self._agg is not None else "not fitted"
        return (
            f"CrowdFeatureExtractor("
            f"red={self.red!r}, N={self.N}, "
            f"n_weeks={self.n_weeks}, "
            f"norm={self.norm!r}, status={fitted})"
        )


# ------------------------------------------------------------------
# Helpers (idénticos a individual_features.py para independencia del módulo)
# ------------------------------------------------------------------

def _add_temporal(
    df: pd.DataFrame,
    slot_minutes: int = 30,
    holiday_dates: frozenset = frozenset(),
) -> pd.DataFrame:
    """
    Añade columnas 'slot' y 'daytype' desde 'timestamp_validacion'.

    slot    : entero 0..(1440//slot_minutes - 1).  Con slot_minutes=30 → 0-47.
    daytype : 'WD' (lun-vie) o 'WE' (sáb-dom + festivos en holiday_dates).
    """
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
