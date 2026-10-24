"""
residence_inferrer.py
=====================
Infiere la parada de residencia de cada usuario a partir de su historial
de validaciones: la estación desde la que el usuario parte con mayor frecuencia
como **primer viaje del día** es la candidata a parada de residencia.

Criterio de confirmación
------------------------
Se confirma la residencia si:
  1. El usuario tiene al menos ``min_days`` días distintos de historial.
  2. La estación más frecuente como primer viaje tiene al menos
     ``dominance_ratio`` veces más ocurrencias que la segunda más frecuente.
     Si solo hay una estación candidata (dominancia absoluta), se confirma
     directamente (equivale a dominance_ratio = ∞).

La parada puede ser de cualquier modo de transporte (metro, bus, cercanías…):
lo que importa es identificar la zona de origen habitual del usuario,
independientemente del modo.

Uso
---
    inferrer = ResidenceInferrer(dominance_ratio=2.0, min_days=5)
    inferrer.fit(history_df)        # history_df: id_tarjeta, punto_acceso, timestamp_validacion
    inferrer.coverage()             # estadísticas de cobertura
    inferrer.get_residence("X123")  # → "02_L1_P10" o None
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas mínimas requeridas en history_df
_REQUIRED = {"id_tarjeta", "punto_acceso", "timestamp_validacion"}


class ResidenceInferrer:
    """
    Infiere la parada de residencia de cada usuario.

    Parameters
    ----------
    dominance_ratio : float
        Ratio mínimo (count_top / count_second) para confirmar residencia.
        Default 2.0. Un valor más alto → más restrictivo (menos usuarios
        confirmados, mayor certeza). Con 1 sola estación se confirma siempre.
    min_days : int
        Número mínimo de días distintos de historial para intentar la
        inferencia. Usuarios con menos días → no confirmados. Default 5.
    """

    def __init__(
        self,
        dominance_ratio: float = 2.0,
        min_days: int = 5,
    ) -> None:
        self.dominance_ratio = dominance_ratio
        self.min_days = min_days

        # Estado interno — solo existe tras fit()
        self._residence: dict[str, str] | None = None   # user → punto_acceso
        self._stats: dict[str, dict]    | None = None   # user → diagnóstico

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, history_df: pd.DataFrame) -> "ResidenceInferrer":
        """
        Infiere la parada de residencia de todos los usuarios en history_df.

        Parameters
        ----------
        history_df : pd.DataFrame
            Historial de validaciones (TODOS los modos).
            Columnas requeridas: id_tarjeta, punto_acceso, timestamp_validacion.

        Returns
        -------
        self
        """
        _validate_columns(history_df, _REQUIRED)

        h = history_df[["id_tarjeta", "punto_acceso", "timestamp_validacion"]].copy()
        h["timestamp_validacion"] = pd.to_datetime(
            h["timestamp_validacion"], utc=False
        ).dt.tz_localize(None)
        h["_fecha"] = h["timestamp_validacion"].dt.date

        # ── Primer viaje del día por usuario ─────────────────────────────
        # idxmin sobre timestamp → una fila por (id_tarjeta, _fecha)
        first_idx = (
            h.groupby(["id_tarjeta", "_fecha"])["timestamp_validacion"]
            .idxmin()
        )
        first_per_day = h.loc[first_idx, ["id_tarjeta", "punto_acceso", "_fecha"]]

        # ── Número de días distintos por usuario ─────────────────────────
        n_days_per_user = (
            first_per_day.groupby("id_tarjeta")["_fecha"]
            .nunique()
            .rename("n_days")
        )

        # ── Frecuencia de cada estación como primer viaje ─────────────────
        freq = (
            first_per_day.groupby(["id_tarjeta", "punto_acceso"])
            .size()
            .reset_index(name="cnt")
        )

        # ── Aplicar criterios por usuario ─────────────────────────────────
        residence: dict[str, str] = {}
        stats:     dict[str, dict] = {}

        for uid, grp in freq.groupby("id_tarjeta"):
            n_days = int(n_days_per_user.get(uid, 0))
            grp_sorted = grp.sort_values("cnt", ascending=False).reset_index(drop=True)
            top_station = grp_sorted.loc[0, "punto_acceso"]
            top_count   = int(grp_sorted.loc[0, "cnt"])
            second_count = int(grp_sorted.loc[1, "cnt"]) if len(grp_sorted) > 1 else 0

            if n_days < self.min_days:
                reason = "low_days"
                confirmed = False
            elif second_count == 0:
                # Dominancia absoluta: solo una estación de primer viaje
                reason = "confirmed"
                confirmed = True
            elif top_count / second_count >= self.dominance_ratio:
                reason = "confirmed"
                confirmed = True
            else:
                reason = "low_dominance"
                confirmed = False

            if confirmed:
                residence[uid] = top_station

            stats[uid] = {
                "station":    top_station,
                "count_1st":  top_count,
                "count_2nd":  second_count,
                "n_days":     n_days,
                "confirmed":  confirmed,
                "reason":     reason,
            }

        self._residence = residence
        self._stats     = stats

        cov = self.coverage()
        logger.info(
            "ResidenceInferrer fit: %d usuarios | %d confirmados (%.1f%%) | "
            "dominance_ratio=%.1f | min_days=%d",
            cov["n_users_total"], cov["n_confirmed"], cov["pct_confirmed"] * 100,
            self.dominance_ratio, self.min_days,
        )
        return self

    def get_residence(self, user_id: str) -> str | None:
        """
        Devuelve la parada de residencia confirmada del usuario, o None.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        str | None
            punto_acceso de la parada de residencia, o None si no confirmada.
        """
        if self._residence is None:
            raise RuntimeError("Llama a fit() antes de get_residence().")
        return self._residence.get(user_id)

    def residence_series(self, users: list[str]) -> pd.Series:
        """
        Devuelve una Series con índice=user_id y valor=punto_acceso de residencia.
        NaN para usuarios sin residencia confirmada o desconocidos.

        Parameters
        ----------
        users : list[str]

        Returns
        -------
        pd.Series, índice=user_id, dtype=object
        """
        if self._residence is None:
            raise RuntimeError("Llama a fit() antes de residence_series().")
        return pd.Series(
            {uid: self._residence.get(uid) for uid in users},
            name="residence_station",
        )

    def coverage(self) -> dict:
        """
        Estadísticas de cobertura de la inferencia.

        Returns
        -------
        dict con claves:
            n_users_total  : int
            n_confirmed    : int
            pct_confirmed  : float  (0-1)
            reason_counts  : dict {reason: count}
        """
        if self._stats is None:
            raise RuntimeError("Llama a fit() antes de coverage().")

        n_total    = len(self._stats)
        n_confirmed = sum(1 for s in self._stats.values() if s["confirmed"])
        reason_counts: dict[str, int] = {}
        for s in self._stats.values():
            r = s["reason"]
            reason_counts[r] = reason_counts.get(r, 0) + 1

        return {
            "n_users_total": n_total,
            "n_confirmed":   n_confirmed,
            "pct_confirmed": n_confirmed / n_total if n_total > 0 else 0.0,
            "reason_counts": reason_counts,
        }

    def __repr__(self) -> str:
        fitted = "fitted" if self._residence is not None else "not fitted"
        n = len(self._residence) if self._residence else 0
        return (
            f"ResidenceInferrer("
            f"dominance_ratio={self.dominance_ratio}, "
            f"min_days={self.min_days}, "
            f"n_confirmed={n}, status={fitted})"
        )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing}")
