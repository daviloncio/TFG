"""
region_function_lda.py
======================
Extrae las "region-function features" de cada estación usando LDA,
tal como se describe en el paper MDLF (Zhao et al., 2022) y en
lda_station_region_guide.txt.

Idea central
------------
Cada estación es un "documento" cuyas "palabras" son patrones
OD-temporales codificados como tokens del tipo:

    OUT_<destino>_k<slot>_<WD|WE>   — viajes salientes
    IN_<origen>_k<slot>_<WD|WE>     — viajes entrantes

LDA descubre K funciones latentes (tópicos). La salida es una
matriz Theta [N_estaciones × K] donde cada fila es el vector de
mezcla de tópicos de esa estación (suma 1 por fila).

Ejemplo de Theta con K=3:

    estacion       topic_0  topic_1  topic_2
    02_L3_P10       0.70     0.20     0.10   ← residencial dominante
    02_L1_P5        0.05     0.85     0.10   ← nodo de intercambio
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {
    "intervalo",
    "origen_punto_acceso",
    "destino_punto_acceso",
    "num_viajes",
}

# Mapeo slot (15 min) → nombre de periodo funcional.
# Cubre los 96 slots del día (0-95).
_SLOT_TO_PERIOD: list[str] = (
    ["noche"] * 24          # slots  0-23  → 00:00-05:45
    + ["madrug"] * 8        # slots 24-31  → 06:00-07:45
    + ["punta_man"] * 8     # slots 32-39  → 08:00-09:45
    + ["mediodia"] * 16     # slots 40-55  → 10:00-13:45
    + ["tarde"] * 12        # slots 56-67  → 14:00-16:45
    + ["punta_tar"] * 12    # slots 68-79  → 17:00-19:45
    + ["noche_ini"] * 16    # slots 80-95  → 20:00-23:45
)

# Mapeo slot (15 min) → 4 bins de 6h (esquema del paper MDLF, H=4).
# Para cambiar el número de bins, modifica esta lista (debe tener 96 entradas).
_SLOT_TO_4BINS: list[str] = (
    ["night"] * 24          # slots  0-23  → 00:00-05:45
    + ["morning"] * 24      # slots 24-47  → 06:00-11:45
    + ["afternoon"] * 24    # slots 48-71  → 12:00-17:45
    + ["evening"] * 24      # slots 72-95  → 18:00-23:45
)


class RegionFunctionLDA:
    """
    LDA sobre patrones OD-temporales para obtener representaciones
    funcionales de estaciones.

    Parameters
    ----------
    n_topics : int
        Número de tópicos latentes K. Empieza con 16.
    slot_minutes : int
        Duración de cada bin temporal en minutos (15 ó 30).
    alpha : float or None
        Prior doc-topic (sklearn: doc_topic_prior).
        Bajo → distribuciones peaky (estaciones especializadas).
        None → sklearn usa 1/n_topics.
    beta : float or None
        Prior topic-word (sklearn: topic_word_prior).
        Bajo → tópicos concentrados en pocos tokens.
        None → sklearn usa 1/n_topics.
    holiday_dates : iterable or None
        Fechas a tratar como fin de semana (festivos). Formato YYYY-MM-DD.
    filter_prefixes : iterable or None
        Si se indica, filtra el OD antes de procesar: solo se conservan filas
        donde TANTO origen COMO destino empiezan por alguno de los prefijos.
        Ejemplo: ['02', '04'] → solo viajes Metro↔Metro, Cercanías↔Cercanías
        y Metro↔Cercanías. Los viajes con bus o interurbano se descartan.
    random_state : int
    max_iter : int
        Iteraciones del EM variacional.
    norm : str or None
        Normalización per-estación aplicada sobre theta tras el LDA.
        'l2', 'l1' o 'max'. None (default) → sin normalización
        (theta es distribución de probabilidad que suma 1, igual que el paper).
    """

    def __init__(
        self,
        n_topics: int = 16,
        slot_minutes: int = 15,
        alpha: float | None = None,
        beta: float | None = None,
        holiday_dates: Iterable | None = None,
        filter_prefixes: Iterable | None = None,
        random_state: int = 42,
        max_iter: int = 20,
        use_periods: bool = False,
        min_df: int = 1,
        log_counts: bool = False,
        df_smoothing: bool = False,
        n_init: int = 1,
        token_granularity: str = "station",
        min_station_trips: int = 0,
        norm: str | None = None,
    ) -> None:
        """
        token_granularity : {'station', 'line', 'temporal'}
            'station'  — comportamiento original: el token incluye el ID completo
                         de la estación destino/origen (OUT_02_L12_P20_morning_WD).
                         Captura relaciones topológicas exactas pero puede crear
                         tópicos ligados a estaciones o líneas concretas.
            'line'     — agrega el destino/origen al nivel de línea extrayendo la
                         parte central del ID (02_L12_P20 → L12).
                         Token resultante: OUT_L12_morning_WD.
                         Útil cuando las líneas tienen función diferenciada;
                         no recomendado para cercanías (todas 04_L0_Pxxx).
            'temporal' — elimina completamente el destino/origen del token.
                         Token resultante: OUT_morning_WD / IN_evening_WE.
                         Vocabulario = 2 × n_bins × n_daytypes (p.ej. 28 tokens
                         con use_periods=True y 2 daytypes).
                         Apropiado para redes con estructura de líneas (metro)
                         donde cualquier token de destino crea tópicos topológicos.
                         LDA aprende perfiles temporales funcionales: residencial
                         (salida mañana, entrada tarde), comercial (inverso), hub
                         (equilibrado todo el día), ocio (actividad en WE), etc.

        min_station_trips : int
            Umbral mínimo de viajes totales (suma de conteos de todos los tokens)
            que debe tener una estación para incluirse en el corpus LDA.
            0 → no filtra (comportamiento original).
            Sirve para excluir estaciones outlier de muy bajo volumen (terminales
            rurales, apeaderos) que de otro modo acaparan un tópico entero.
        """
        self.n_topics = n_topics
        self.slot_minutes = slot_minutes
        self.alpha = alpha
        self.beta = beta
        self.filter_prefixes: list[str] | None = list(filter_prefixes) if filter_prefixes is not None else None
        self.random_state = random_state
        self.max_iter = max_iter
        self.use_periods = use_periods
        self.min_df = min_df
        self.log_counts = log_counts
        self.df_smoothing = df_smoothing
        self.n_init = n_init
        self.token_granularity = token_granularity
        self.min_station_trips = min_station_trips
        self.norm = norm

        self.holiday_dates: set = set()
        if holiday_dates is not None:
            self.holiday_dates = {pd.Timestamp(d).date() for d in holiday_dates}

        # Estado interno — solo existe tras fit
        self.vectorizer_: DictVectorizer | None = None
        self.lda_: LatentDirichletAllocation | None = None
        self.station_index_: list[str] | None = None
        self.theta_: pd.DataFrame | None = None
        self.X_: sp.csr_matrix | None = None   # matriz doc-term usada en el último fit

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, od_df: pd.DataFrame) -> "RegionFunctionLDA":
        """
        Aprende el vocabulario y ajusta el modelo LDA.

        Internamente:
          1. build_documents → dict {estacion: {token: count}}
          2. DictVectorizer.fit_transform → sparse matrix X [estaciones × tokens]
          3. LatentDirichletAllocation.fit(X)

        Parameters
        ----------
        od_df : pd.DataFrame
            Una o más matrices OD concatenadas.
            Columnas requeridas: intervalo, origen_punto_acceso,
                                 destino_punto_acceso, num_viajes.

        Returns
        -------
        self
        """
        self._validate(od_df)
        od_df = self._filter(od_df)
        docs = self.build_documents(od_df)
        X = self._vectorize_fit(docs)
        self._init_lda()
        self._fit_best(X)
        logger.info(
            "LDA ajustado: %d estaciones | %d tokens | %d tópicos",
            X.shape[0], X.shape[1], self.n_topics,
        )
        return self

    def transform(self, od_df: pd.DataFrame) -> pd.DataFrame:
        """
        Proyecta nuevos datos OD sobre los tópicos ya aprendidos.

        Tokens no vistos durante el fit se ignoran silenciosamente.
        Útil para evaluar estaciones de un período distinto al de entrenamiento.

        Parameters
        ----------
        od_df : pd.DataFrame
            Misma estructura que la entrada de fit().

        Returns
        -------
        theta : pd.DataFrame  [estaciones × n_topics]
            Cada fila es la distribución de tópicos de esa estación.
        """
        self._check_fitted()
        self._validate(od_df)
        od_df = self._filter(od_df)
        docs = self.build_documents(od_df)
        X = self._vectorize_transform(docs)
        theta_arr = self.lda_.transform(X)
        return self._to_theta(theta_arr, self.station_index_)

    def fit_transform(self, od_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta el modelo y devuelve Theta en un único paso.

        Parameters
        ----------
        od_df : pd.DataFrame

        Returns
        -------
        theta : pd.DataFrame  [estaciones × n_topics]
        """
        self._validate(od_df)
        od_df = self._filter(od_df)
        docs = self.build_documents(od_df)
        X = self._vectorize_fit(docs)
        self._init_lda()
        self._fit_best(X)
        theta_arr = self.lda_.transform(X)
        self.theta_ = self._to_theta(theta_arr, self.station_index_)
        logger.info(
            "fit_transform: %d estaciones × %d tópicos (vocab: %d tokens)",
            len(self.station_index_), self.n_topics,
            len(self.vectorizer_.get_feature_names_out()),
        )
        return self.theta_

    def build_documents(self, od_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Construye un documento por estación a partir de los flujos OD.

        Cada documento es un bag-of-words donde:
          - palabra  = patrón OD-temporal codificado como token
          - frecuencia = número de viajes que representa ese patrón

        Tokens generados:
          OUT_<destino>_k<slot>_<WD|WE>  → en el documento del ORIGEN
          IN_<origen>_k<slot>_<WD|WE>   → en el documento del DESTINO

        Implementación vectorizada: no itera filas — usa groupby + concat.

        Parameters
        ----------
        od_df : pd.DataFrame

        Returns
        -------
        documents : dict[str, dict[str, float]]
            {id_estacion: {token: count_acumulado}}
        """
        df = self._prepare(od_df)

        # Eliminar self-trips: viajes donde origen == destino (datos sucios o error de predicción)
        n_before = len(df)
        df = df[df["origen_punto_acceso"] != df["destino_punto_acceso"]]
        if len(df) < n_before:
            logger.info(
                "build_documents: eliminados %d self-trips (origen == destino)",
                n_before - len(df),
            )

        # Columna de agrupación temporal:
        #   use_periods=True → 7 periodos nombrados (_SLOT_TO_PERIOD)
        #   use_periods=4    → 4 bins de 6h (_SLOT_TO_4BINS, esquema paper MDLF)
        #   use_periods=False → slot numérico raw (96 bins con slot_minutes=15)
        if self.use_periods is True:
            df["time_bin"] = df["slot"].map(lambda s: _SLOT_TO_PERIOD[s])
        elif self.use_periods == 4:
            df["time_bin"] = df["slot"].map(lambda s: _SLOT_TO_4BINS[s])
        else:
            df["time_bin"] = "k" + df["slot"].astype(str).str.zfill(2)

        # Agregar por (origen, destino, time_bin, daytype) sumando viajes.
        # Esto colapsa múltiples días al mismo bin: clave para LDA histórico.
        agg = (
            df.groupby(
                ["origen_punto_acceso", "destino_punto_acceso", "time_bin", "daytype"],
                observed=True,
                sort=False,
            )["num_viajes"]
            .sum()
            .reset_index()
        )

        # Construir strings de token vectorialmente.
        #
        # token_granularity='station'  → token incluye ID completo de estación destino/origen
        #                                 OUT_02_L12_P20_morning_WD
        # token_granularity='line'     → agrega destino/origen al nivel de línea
        #                                 OUT_L12_morning_WD
        # token_granularity='temporal' → elimina completamente el destino/origen del token;
        #                                 el conteo es el total de viajes en esa
        #                                 dirección/franja/tipodía sin importar adónde van.
        #                                 OUT_morning_WD  /  IN_evening_WE
        #                                 Vocabulario = 2*n_bins*n_daytypes tokens.
        #                                 Apropiado para redes con estructura de líneas
        #                                 (metro) donde la topología dominaría cualquier
        #                                 token basado en destino. LDA aprende perfiles
        #                                 temporales funcionales: residencial, comercial,
        #                                 hub, ocio, etc.
        if self.token_granularity == "temporal":
            agg["out_token"] = "OUT_" + agg["time_bin"] + "_" + agg["daytype"]
            agg["in_token"]  = "IN_"  + agg["time_bin"] + "_" + agg["daytype"]
        elif self.token_granularity == "line":
            dest_key = agg["destino_punto_acceso"].str.split("_").str[1]
            orig_key = agg["origen_punto_acceso"].str.split("_").str[1]
            agg["out_token"] = "OUT_" + dest_key + "_" + agg["time_bin"] + "_" + agg["daytype"]
            agg["in_token"]  = "IN_"  + orig_key + "_" + agg["time_bin"] + "_" + agg["daytype"]
        else:  # 'station'
            agg["out_token"] = "OUT_" + agg["destino_punto_acceso"] + "_" + agg["time_bin"] + "_" + agg["daytype"]
            agg["in_token"]  = "IN_"  + agg["origen_punto_acceso"]  + "_" + agg["time_bin"] + "_" + agg["daytype"]

        # Construir dos DataFrames largos: (station, token, count)
        out_df = agg[["origen_punto_acceso", "out_token", "num_viajes"]].rename(
            columns={"origen_punto_acceso": "station", "out_token": "token", "num_viajes": "count"}
        )
        in_df = agg[["destino_punto_acceso", "in_token", "num_viajes"]].rename(
            columns={"destino_punto_acceso": "station", "in_token": "token", "num_viajes": "count"}
        )

        # Unir OUT e IN y agregar por (station, token)
        combined = (
            pd.concat([out_df, in_df], ignore_index=True)
            .groupby(["station", "token"], observed=True, sort=False)["count"]
            .sum()
        )

        # Convertir MultiIndex Series a dict de dicts
        documents: dict[str, dict[str, float]] = {}
        for station, group in combined.groupby(level=0):
            raw = group.droplevel(0)
            documents[station] = (np.log1p(raw) if self.log_counts else raw).to_dict()

        # Filtrar estaciones con muy pocos viajes totales (outliers de bajo volumen)
        if self.min_station_trips > 0:
            n_before = len(documents)
            documents = {
                s: doc for s, doc in documents.items()
                if sum(doc.values()) >= self.min_station_trips
            }
            n_removed = n_before - len(documents)
            if n_removed:
                logger.info(
                    "min_station_trips=%d: eliminadas %d estaciones de bajo volumen "
                    "(%d → %d estaciones)",
                    self.min_station_trips, n_removed, n_before, len(documents),
                )

        return documents

    # ------------------------------------------------------------------
    # Inspección de tópicos
    # ------------------------------------------------------------------

    def inspect_topics(self, n_top_tokens: int = 15) -> pd.DataFrame:
        """
        Devuelve los tokens más representativos de cada tópico.

        Internamente usa lda_.components_ [n_topics × n_tokens]:
        cada fila es la distribución de palabras (no normalizada) del tópico.
        Los tokens con mayor peso son los que "definen" ese tópico.

        Cómo leer la salida
        -------------------
        Si un tópico tiene tokens dominantes como:
          OUT_*_k14_WD, OUT_*_k15_WD, IN_*_k34_WD, IN_*_k35_WD
        → pico de salida a las 7h y entrada a las 17h en laborable
        → probable función RESIDENCIAL

        Si los tokens dominantes son:
          IN_*_k14_WD, IN_*_k15_WD, OUT_*_k34_WD, OUT_*_k35_WD
        → pico de entrada a las 7h y salida a las 17h en laborable
        → probable función COMERCIAL / OFICINAS

        Parameters
        ----------
        n_top_tokens : int
            Número de tokens a mostrar por tópico.

        Returns
        -------
        pd.DataFrame  con columnas [top_tokens, top_weights]
        """
        self._check_fitted()
        feature_names = np.array(self.vectorizer_.get_feature_names_out())
        rows = []
        for topic_idx, weights in enumerate(self.lda_.components_):
            # Ordenar tokens de mayor a menor peso
            top_idx = weights.argsort()[::-1][:n_top_tokens]
            top_tokens = feature_names[top_idx].tolist()
            # Normalizar a probabilidad dentro del tópico
            top_weights = (weights[top_idx] / weights.sum()).tolist()
            rows.append({
                "topic": f"topic_{topic_idx}",
                "top_tokens": top_tokens,
                "top_weights": [round(w, 5) for w in top_weights],
            })
        return pd.DataFrame(rows).set_index("topic")

    def topic_summary(self, n_top_tokens: int = 10) -> str:
        """
        Resumen legible por consola: un tópico por línea con sus tokens principales.

        Ejemplo de salida:
            topic_0 : IN_02_L3_P9_k16_WD(0.031) | OUT_02_L1_P5_k34_WD(0.028) | ...
            topic_1 : OUT_02_L3_P9_k14_WD(0.045) | ...
        """
        df = self.inspect_topics(n_top_tokens)
        lines = []
        for topic, row in df.iterrows():
            pairs = [
                f"{t}({w:.4f})"
                for t, w in zip(row["top_tokens"], row["top_weights"])
            ]
            lines.append(f"{topic} : {' | '.join(pairs)}")
        return "\n".join(lines)

    def station_profile(self, station_id: str) -> pd.Series:
        """
        Devuelve el vector de tópicos de una estación concreta.

        Parameters
        ----------
        station_id : str
            Identificador de estación, p.ej. '02_L3_P10'.

        Returns
        -------
        pd.Series  con índice [topic_0, topic_1, ...]
        """
        self._check_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Theta no está en memoria. Usa fit_transform() en lugar de fit() + transform()."
            )
        if station_id not in self.theta_.index:
            raise KeyError(f"Estación '{station_id}' no encontrada en Theta.")
        return self.theta_.loc[station_id]

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _prepare(self, od_df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade columnas slot y daytype al dataframe.

        Los timestamps de las matrices OD ya están en hora de Madrid.
        El +00:00 que aparece en matrices recientes es solo una etiqueta
        añadida al guardar, no indica UTC real — no se convierte timezone.

        El día virtual (corte a las 4am) no afecta al slot: para LDA
        lo que importa es la hora del día en que ocurre el viaje, no
        a qué día operativo pertenece.
        """
        df = od_df.copy()

        # Parsear ignorando timezone (ya está en hora local Madrid)
        ts = pd.to_datetime(df["intervalo"], utc=False).dt.tz_localize(None)

        # Slot: índice entero dentro del día
        # Ejemplo slot_minutes=15: 04:00 → slot 16, 08:30 → slot 34
        df["slot"] = (ts.dt.hour * 60 + ts.dt.minute) // self.slot_minutes

        # Tipo de día: fin de semana o festivo → WE, resto → WD
        day_of_week = ts.dt.dayofweek   # 0=lunes ... 6=domingo
        date_part   = ts.dt.date
        is_holiday  = date_part.apply(lambda d: d in self.holiday_dates)
        df["daytype"] = np.where((day_of_week >= 5) | is_holiday, "WE", "WD")

        return df

    def _filter_vocab(self, documents: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """
        Elimina tokens que aparecen en menos de self.min_df estaciones distintas
        y, opcionalmente, aplica df_smoothing: pondera cada token por la fracción
        de estaciones que lo usan.

        df_smoothing=True:
            weight(s, t) = count(s, t) * df(t) / N_estaciones

            Los tokens topológicos (presentes en 1-2 estaciones como
            OUT_BAMBÚ_k08_WD) quedan casi a cero.
            Los tokens funcionales (presentes en muchas estaciones como
            OUT_SOL_k08_WD) mantienen su peso.
            Esto evita que LDA cree tópicos asociados a posición en la red.
        """
        from collections import Counter
        station_freq: Counter = Counter()
        for doc in documents.values():
            station_freq.update(doc.keys())

        # Filtro min_df
        if self.min_df > 1:
            keep = {token for token, freq in station_freq.items() if freq >= self.min_df}
            n_before = sum(len(d) for d in documents.values())
            documents = {
                s: {t: c for t, c in doc.items() if t in keep}
                for s, doc in documents.items()
            }
            n_after = sum(len(d) for d in documents.values())
            logger.info(
                "min_df=%d: vocabulario %d → %d tokens únicos (docs: %d → %d entradas)",
                self.min_df, len(station_freq), len(keep), n_before, n_after,
            )
            # Recalcular frecuencias tras el filtro
            station_freq = Counter()
            for doc in documents.values():
                station_freq.update(doc.keys())

        # df_smoothing: ponderar cada token por su fracción de cobertura
        if self.df_smoothing:
            N = len(documents)
            documents = {
                s: {t: c * (station_freq[t] / N) for t, c in doc.items()}
                for s, doc in documents.items()
            }
            logger.info("df_smoothing aplicado: tokens ponderados por df/N_estaciones")

        return documents

    def _vectorize_fit(self, documents: dict[str, dict[str, float]]) -> sp.csr_matrix:
        """
        Aprende vocabulario y convierte documentos en matriz dispersa.

        DictVectorizer toma una lista de dicts {token: count} y construye
        automáticamente el vocabulario y la matriz sparse X.
        """
        stations = sorted(documents.keys())
        self.station_index_ = stations
        documents = self._filter_vocab(documents)
        self.vectorizer_ = DictVectorizer(sparse=True, dtype=np.float64)
        X = self.vectorizer_.fit_transform([documents[s] for s in stations])
        self.X_ = X
        return X

    def _vectorize_transform(self, documents: dict[str, dict[str, float]]) -> sp.csr_matrix:
        """
        Vectoriza nuevos documentos usando el vocabulario ya aprendido.

        Tokens desconocidos (no vistos en fit) se ignoran silenciosamente:
        DictVectorizer.transform() los descarta y pone 0 en esa columna.
        Esto es el comportamiento correcto para datos nuevos.
        """
        stations = sorted(documents.keys())
        self.station_index_ = stations
        return self.vectorizer_.transform([documents[s] for s in stations])

    def _init_lda(self) -> None:
        """Instancia el modelo sklearn con los hiperparámetros actuales."""
        self.lda_ = LatentDirichletAllocation(
            n_components=self.n_topics,
            doc_topic_prior=self.alpha,        # alpha
            topic_word_prior=self.beta,        # beta
            learning_method="batch",           # batch = más estable para nuestro tamaño
            max_iter=self.max_iter,
            evaluate_every=-1,                 # desactivar cómputo de perplexity en cada iter
            random_state=self.random_state,
        )

    def _fit_best(self, X: sp.csr_matrix) -> None:
        """
        Entrena LDA n_init veces con distintas semillas y conserva el mejor
        modelo según log-likelihood sobre los datos de entrenamiento.

        Con n_init=1 (defecto) el comportamiento es idéntico al original.
        Con n_init>1 se reduce el riesgo de tópicos vacíos por mala
        inicialización: LDA puede converger a mínimos locales distintos
        y algunos producen tópicos muertos; quedarse con el de mayor
        log-likelihood selecciona la solución más informativa.
        """
        if self.n_init == 1:
            self.lda_.fit(X)
            return

        best_score = -np.inf
        best_lda = None
        for i in range(self.n_init):
            lda = LatentDirichletAllocation(
                n_components=self.n_topics,
                doc_topic_prior=self.alpha,
                topic_word_prior=self.beta,
                learning_method="batch",
                max_iter=self.max_iter,
                evaluate_every=-1,
                random_state=self.random_state + i,
            )
            lda.fit(X)
            score = lda.score(X)
            logger.info("n_init %d/%d — log-likelihood: %.2f", i + 1, self.n_init, score)
            if score > best_score:
                best_score = score
                best_lda = lda

        self.lda_ = best_lda
        logger.info("Mejor inicialización: log-likelihood=%.2f", best_score)

    def _to_theta(self, theta_arr: np.ndarray, stations: list[str]) -> pd.DataFrame:
        """Convierte el array numpy de LDA en DataFrame con nombres de estaciones."""
        if self.norm:
            theta_arr = normalize(theta_arr, norm=self.norm, axis=1)
        cols = [f"topic_{i}" for i in range(self.n_topics)]
        return pd.DataFrame(theta_arr, index=stations, columns=cols)

    def _filter(self, od_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra el OD para conservar solo filas donde origen Y destino
        empiezan por alguno de los prefijos en self.filter_prefixes.
        Si filter_prefixes es None, devuelve el dataframe sin tocar.
        """
        if not self.filter_prefixes:
            return od_df
        prefixes = tuple(self.filter_prefixes)
        mask = (
            od_df['origen_punto_acceso'].str.startswith(prefixes) &
            od_df['destino_punto_acceso'].str.startswith(prefixes)
        )
        filtrado = od_df[mask]
        logger.info(
            "Filtro prefijos %s: %d → %d filas (%.1f%%)",
            self.filter_prefixes, len(od_df), len(filtrado),
            len(filtrado) / len(od_df) * 100,
        )
        return filtrado

    def _validate(self, od_df: pd.DataFrame) -> None:
        """Comprueba que el DataFrame tiene las columnas mínimas necesarias."""
        missing = _REQUIRED_COLS - set(od_df.columns)
        if missing:
            raise ValueError(
                f"OD DataFrame faltan columnas: {missing}. "
                f"Columnas presentes: {list(od_df.columns)}"
            )
        if od_df.empty:
            raise ValueError("El DataFrame de entrada está vacío.")

    def _check_fitted(self) -> None:
        """Lanza error si el modelo no ha sido ajustado todavía."""
        if self.lda_ is None or self.vectorizer_ is None:
            raise RuntimeError(
                "Modelo no ajustado. Llama a fit() o fit_transform() primero."
            )

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = "fitted" if self.lda_ is not None else "not fitted"
        return (
            f"RegionFunctionLDA("
            f"n_topics={self.n_topics}, "
            f"slot_minutes={self.slot_minutes}, "
            f"use_periods={self.use_periods}, "
            f"token_granularity={self.token_granularity!r}, "
            f"min_station_trips={self.min_station_trips}, "
            f"min_df={self.min_df}, "
            f"log_counts={self.log_counts}, "
            f"df_smoothing={self.df_smoothing}, "
            f"n_init={self.n_init}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"norm={self.norm!r}, "
            f"filter_prefixes={self.filter_prefixes}, "
            f"status={fitted})"
        )
