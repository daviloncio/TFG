"""
pipeline_mdlf.py
================
Sub-pipeline MDLF para predicción de destino de viajes UNLINKED de cercanías.

Se lanza desde cubo_izquierdo/pipeline.py cuando 20 in hipotesis.

Flujo
-----
1. _build_labeled_set : UNLINKED cercanías ∩ crear_mapeos → labeled_c + destino_real
2. _split             : train/test split estratificado
3. _fit_lda           : RegionFunctionLDA sobre matrices OD → theta_c [N × K]
4. _load_line         : carga line_cercanias_ap.parquet → line_c [N × 64]
5. _extract_features  : IndividualFeatureExtractor + CrowdFeatureExtractor
6. _build_V           : ODRegionCorrelation → F'_R; ensambla V ∈ R^(n, N, D)
7. _train_model       : bucle de épocas + batches (CNN pendiente de implementar)
8. _evaluate          : accuracy top-1 / top-k en test

Dimensiones de features por destino s_i
-----------------------------------------
  F_u[i]    : 6  sub-features individuales (8 con include_prox=True)  (IndividualFeatureExtractor)
  F_crowd[i]: n_weeks crowd features                                   (CrowdFeatureExtractor)
  F'_R[i]   : H  scores OD correlation heads                          (_ODRegionModule)
  ──────────────────────────────────────────────
  v_i       : D_ind + n_weeks + H  →  D dims  (D_ind=6 por defecto)

LDA (K) y LINE (64) se usan para calcular v_o y v_d en ODRegionCorrelation,
pero NO entran directamente en V.

Entrada CNN por viaje: V ∈ R^(N × D) — una fila por destino posible.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths por defecto (relativos al directorio de trabajo del pipeline principal)
# ---------------------------------------------------------------------------
_ROOT            = Path(__file__).parent.parent           # david/estudios/
_RUNS_DIR        = Path(__file__).parent / "runs"         # MDLF/runs/


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from tee_logger import Tee as _Tee
_OD_MATRICES_DIR = _ROOT / "cubo_izquierdo" / "matrices"
_PARADAS_VECINAS = _ROOT / "topologia" / "paradas_vecinas_1500m.csv"
_LINE_AP         = _ROOT / "MDLF" / "LINE" / "output" / "line_cercanias_ap.parquet"
_CROWD_FILE           = _ROOT / "MDLF" / "CROWD FEATURES" / "crowd_features.py"
_RESIDENCE_FILE       = _ROOT / "MDLF" / "CROWD FEATURES" / "residence_inferrer.py"
_NBCROWD_FILE         = _ROOT / "MDLF" / "CROWD FEATURES" / "neighborhood_crowd.py"


# ---------------------------------------------------------------------------
# CNN — placeholder
# ---------------------------------------------------------------------------

class ODRegionCorrelation:
    """
    O-D Region Correlation Learning — Section V.A (Zhao et al., 2022).

    Para cada viaje calcula F'_R ∈ ℝ^(N×H) mediante atención bilineal
    multi-head entre la región de origen y las regiones de todos los destinos
    candidatos, modulada por el contexto temporal del viaje.

    Fórmula por head h (v_o, v_d son Region Features de origen y destino):

        W^h = W_1^h · E · W_2^h        ∈ ℝ^(D_R × D_R)
              (D_R×1)(1×D_T)(D_T×D_R)

        score_{h,i} = v_o · W^h · v_{d_i}^T  → escalar por destino candidato i

    Descomposición eficiente por batch:

        a_h     = V_o  @ W_1^h          ∈ ℝ^(n×1)      origen → escalar
        B_h     = V_D  @ W_2^h.T        ∈ ℝ^(N×D_T)    destinos → espacio temporal
        C_h     = E_batch @ B_h.T       ∈ ℝ^(n×N)      modulación temporal × destino
        scores_h = a_h * C_h            ∈ ℝ^(n×N)      [broadcast (n,1)×(n,N)]

    Contexto temporal: tabla de embedding conjunta para (daytype, slot).
        daytype ∈ {0=laborable, 1=festivo/finde}
        slot    ∈ {0…47}  (intervalos de 30 min)
        índice  = daytype * n_slots + slot  ∈ {0…95}
        E       = E_table[índice] ∈ ℝ^(1×D_T)   — inicializada aleatoriamente, entrenable

    F'_R = stack([scores_1, …, scores_H], axis=2) ∈ ℝ^(n, N, H)

    Parameters
    ----------
    D_R : int
        Dimensión de Region Features (K_lda + 64_line).
    D_T : int
        Dimensión del embedding temporal.
    n_heads : int
        Número H de cabezas de atención.
    n_slots : int
        Slots de 30 min por día (default 48 = 24 h × 2).
    n_daytypes : int
        Tipos de día (default 2: laborable / festivo-finde).
    random_state : int
    """

    N_DAYTYPES = 2    # WD / WE-festivo

    def __init__(
        self,
        D_R          : int,
        D_T          : int,
        n_heads      : int,
        slot_minutes : int = 30,
        n_daytypes   : int = 2,
        random_state : int = 42,
        holiday_dates: frozenset = frozenset(),
    ) -> None:
        self.D_R          = D_R
        self.D_T          = D_T
        self.H            = n_heads
        self.slot_minutes = slot_minutes
        self.n_slots      = 1440 // slot_minutes   # slots por día
        self.n_daytypes   = n_daytypes
        self.holiday_dates = frozenset(holiday_dates)
        n_entries         = n_daytypes * self.n_slots

        rng = np.random.default_rng(random_state)

        # Tabla de embedding temporal conjunta: (n_entries, D_T)
        self.E_table = rng.normal(
            0, 1.0 / np.sqrt(D_T), (n_entries, D_T)
        ).astype(np.float32)

        # Pesos bilineales por head
        scale_w = 1.0 / np.sqrt(D_R)
        self.W1 = rng.normal(
            0, scale_w, (n_heads, D_R, 1)
        ).astype(np.float32)                            # [H, D_R, 1]
        self.W2 = rng.normal(
            0, scale_w, (n_heads, D_T, D_R)
        ).astype(np.float32)                            # [H, D_T, D_R]

    def temporal_index(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula índice conjunto (daytype * n_slots + slot) por viaje.

        Returns
        -------
        idx : np.ndarray, shape (n,), dtype int32
        """
        ts      = pd.to_datetime(df["timestamp_validacion"])
        slots_per_hour = 60 // self.slot_minutes
        slot    = (ts.dt.hour * slots_per_hour + ts.dt.minute // self.slot_minutes).astype(int)
        is_we   = (ts.dt.dayofweek >= 5) | ts.dt.date.map(lambda d: d in self.holiday_dates)
        daytype = is_we.astype(int)                                     # 0=WD, 1=WE
        return (daytype * self.n_slots + slot).values.astype(np.int32)

    def compute_batch(
        self,
        v_o         : np.ndarray,   # (n, D_R) — region features del origen por viaje
        V_D         : np.ndarray,   # (N, D_R) — region features de todos los destinos
        temporal_idx: np.ndarray,   # (n,)     — índice de embedding temporal por viaje
    ) -> np.ndarray:
        """
        Calcula F'_R para un batch de n viajes.

        Returns
        -------
        F_prime_R : np.ndarray, shape (n, N, H)
        """
        E_batch   = self.E_table[temporal_idx]           # (n, D_T)
        F_prime_R = np.empty(
            (v_o.shape[0], V_D.shape[0], self.H), dtype=np.float32
        )

        for h in range(self.H):
            a_h = v_o @ self.W1[h]           # (n, D_R) @ (D_R, 1) → (n, 1)
            B_h = V_D @ self.W2[h].T         # (N, D_R) @ (D_R, D_T) → (N, D_T)
            C_h = E_batch @ B_h.T            # (n, D_T) @ (D_T, N)  → (n, N)
            F_prime_R[:, :, h] = a_h * C_h  # (n, 1)  *  (n, N)    → (n, N)

        return F_prime_R


class _ODRegionModule(nn.Module):
    """
    Versión PyTorch entrenable de ODRegionCorrelation.

    Tabla de embedding temporal (nn.Embedding) + pesos bilineales W1, W2
    como nn.Parameter — todos actualizables por backpropagation.

    Parameters
    ----------
    D_R      : dimensión de Region Features (K_lda + 64_line)
    D_T      : dimensión del embedding temporal
    n_heads  : H — número de cabezas de atención bilineal
    n_entries: tamaño de la tabla conjunta (n_daytypes × n_slots)
    """

    def __init__(self, D_R: int, D_T: int, n_heads: int, n_entries: int) -> None:
        super().__init__()
        self.H = n_heads

        # Tabla de embedding temporal conjunta — índice = daytype * n_slots + slot
        self.E_table = nn.Embedding(n_entries, D_T)

        # Pesos bilineales por head (paper-faithful, rank-1):
        #   W^h = W1^h · E · W2^h  ∈ ℝ^(D_R × D_R)
        #   W1^h ∈ ℝ^(D_R×1),  E ∈ ℝ^(1×D_T),  W2^h ∈ ℝ^(D_T×D_R)
        self.W1 = nn.Parameter(torch.randn(n_heads, D_R, 1)    * (D_R ** -0.5))
        self.W2 = nn.Parameter(torch.randn(n_heads, D_T, D_R)  * (D_R ** -0.5))

    def forward(
        self,
        v_o         : torch.Tensor,   # (batch, D_R)
        V_D         : torch.Tensor,   # (N, D_R)   — region features de todos los destinos
        temporal_idx: torch.Tensor,   # (batch,) long
    ) -> torch.Tensor:
        """
        Retorna F'_R ∈ ℝ^(batch, N, H).

        Context features del viaje:
          · Temporal (F^t_c): E = E_table[daytype*48+slot] ∈ ℝ^(D_T)  — embedding entrenable
          · Espacial (v_o):   region attributes de la estación origen   ∈ ℝ^(D_R)

        Por head h (paper-faithful, rank-1 bilineal):
            W^h = W1^h · E · W2^h  ∈ ℝ^(D_R × D_R)
                  W1^h ∈ ℝ^(D_R×1),  E ∈ ℝ^(1×D_T),  W2^h ∈ ℝ^(D_T×D_R)

            Implementación eficiente (descomposición rank-1):
            a_h    = v_o  @ W1[h]      ∈ ℝ^(batch, 1)    — escalar origen
            B_h    = V_D  @ W2[h].T    ∈ ℝ^(N, D_T)      — proyección destinos
            C_h    = E    @ B_h.T      ∈ ℝ^(batch, N)     — modulación temporal × destino
            scores = a_h  * C_h        ∈ ℝ^(batch, N)     — correlación OD
        """
        E = self.E_table(temporal_idx)          # (batch, D_T)
        parts = []
        for h in range(self.H):
            a_h    = v_o @ self.W1[h]           # (batch, D_R) @ (D_R, 1)   → (batch, 1)
            B_h    = V_D @ self.W2[h].T         # (N, D_R)    @ (D_R, D_T)  → (N, D_T)
            C_h    = E   @ B_h.T                # (batch, D_T) @ (D_T, N)   → (batch, N)
            scores = a_h * C_h                  # (batch, 1)  * (batch, N)  → (batch, N)
            parts.append(scores.unsqueeze(2))   # (batch, N, 1)
        return torch.cat(parts, dim=2)          # (batch, N, H)


class _DestinationCNN(nn.Module):
    """
    CNN de predicción de destino — MDLF paper (Zhao et al., 2022), Section V.B.

    Arquitectura:
      1. _ODRegionModule (bilineal multi-head) → F'_R ∈ ℝ^(batch, N, H)
      2. V = [V_static ‖ F'_R]  ∈ ℝ^(batch, N, D)    D = D_static + H
      3. 2 × Conv1d(C canales, kernel M) + ReLU       parámetros compartidos entre destinos
      4. Conv1d(1 canal, kernel M)  → U ∈ ℝ^(batch, N)  score por destino
      5. Destino predicho = argmax(U)   ||   SoftMax(U) para probabilidades

    Cada fila de V (correspondiente a un destino candidato s_i) se procesa con
    los mismos parámetros Θ — equivalente a Conv1d con kernel_size=1 aplicado
    posición a posición, o a kernel > 1 para capturar contexto entre estaciones vecinas.

    Loss: CrossEntropyLoss (incluye SoftMax implícito).

    Parameters
    ----------
    N          : int   Número de estaciones candidatas
    D_static   : int   Dim de features estáticas [Individual ‖ Crowd]
    D_R        : int   Dim de Region Features para la atención bilineal (K + 64)
    D_T        : int   Dim del embedding temporal
    n_heads    : int   H — cabezas de atención bilineal
    n_entries  : int   Entradas de la tabla temporal (n_daytypes × n_slots)
    C          : int   Canales CNN. Default 16 (paper)
    M          : int   Tamaño de kernel. Default 1 (por estación independiente).
                       M=3 añade contexto de estaciones vecinas en el ranking.
                       M debe ser impar para preservar longitud N con padding=(M-1)//2.
    """

    def __init__(
        self,
        N        : int,
        D_static : int,
        D_R      : int,
        D_T      : int,
        n_heads  : int,
        n_entries: int,
        C        : int = 16,
        M        : int = 1,
    ) -> None:
        super().__init__()
        self.N      = N
        self.H      = n_heads   # 0 → OD module desactivado (ablación region features)

        if n_heads > 0:
            self.od_module = _ODRegionModule(D_R, D_T, n_heads, n_entries)
        else:
            self.od_module = None

        D   = D_static + n_heads     # dim completa por estación tras concat
        pad = (M - 1) // 2

        self.conv1 = nn.Conv1d(D, C, kernel_size=M, padding=pad)
        self.conv2 = nn.Conv1d(C, C, kernel_size=M, padding=pad)
        self.conv3 = nn.Conv1d(C, 1, kernel_size=M, padding=pad)

    def forward(
        self,
        V_static    : torch.Tensor,   # (batch, N, D_static)
        v_o         : torch.Tensor,   # (batch, D_R)
        V_D         : torch.Tensor,   # (N, D_R)
        temporal_idx: torch.Tensor,   # (batch,) long
    ) -> torch.Tensor:
        """Retorna logits (raw scores) ∈ ℝ^(batch, N) — pre-SoftMax."""
        if self.od_module is not None:
            F_prime = self.od_module(v_o, V_D, temporal_idx)  # (batch, N, H)
            V = torch.cat([V_static, F_prime], dim=2)          # (batch, N, D)
        else:
            V = V_static                                        # sin OD module

        x = V.permute(0, 2, 1)                                # (batch, D, N)
        x = F.relu(self.conv1(x))[..., :self.N]              # (batch, C, N) — trim por si M par
        x = F.relu(self.conv2(x))[..., :self.N]
        x = self.conv3(x)[..., :self.N].squeeze(1)            # (batch, N)
        return x

    def predict_topk(
        self,
        V_static    : torch.Tensor,
        v_o         : torch.Tensor,
        V_D         : torch.Tensor,
        temporal_idx: torch.Tensor,
        k           : int = 1,
    ) -> torch.Tensor:
        """Índices de los k destinos más probables por viaje → (batch, k)."""
        with torch.no_grad():
            logits = self.forward(V_static, v_o, V_D, temporal_idx)
        return torch.topk(logits, k, dim=1).indices


# ---------------------------------------------------------------------------
# PipelineMDLF
# ---------------------------------------------------------------------------

_MDLF_ALMACEN = Path(__file__).parent / "almacen"


class PipelineMDLF:
    """
    Sub-pipeline de predicción de destino MDLF para cercanías.

    Parameters
    ----------
    nombre_almacen : str
        Nombre de la subcarpeta en MDLF/almacen/ que contiene los 5 parquets
        generados por hipótesis 21 del pipeline de cubo izquierdo:
          - x_pred_checked.parquet
          - unknown.parquet
          - x_test.parquet
          - y_test.parquet
          - subidas_sin_pred.parquet  (vacío si no aplica)
        Ejemplo: "200mil_tarjetas_3_al_5_marzo"
    od_matrices_dir : Path
        Directorio raíz con subcarpetas de matrices OD. Default: cubo_izquierdo/matrices.
    od_matrices_subdirs : list[str] | None
        Subcarpetas exactas a usar para LDA (nombres relativos a od_matrices_dir).
        Si es None, se usan todas las subcarpetas encontradas recursivamente.
        Ejemplo: ["validaciones_3_7_febrero", "validaciones_8_9_febrero", ...]
    hist_individual_paths : list[Path] | None
        Obsoleto — ignorado si hay caché de IndividualHistBuilder en el almacén.
    ttp_paths : list[str] | None
        Rutas a los TTP CSVs de los 6 períodos (~81 GB total).
        Si se define, IndividualHistBuilder los procesa para construir el historial.
        Si es None y no hay caché, se usa x_pred_checked como fallback.
    cubo_hiperparams : dict | None
        Hiperparámetros del pipeline cubo izquierdo (dia_virtual, radio_busqueda, ...).
        Necesarios si ttp_paths está definido y no hay caché.
    paradas_vecinas_path : Path
        CSV de paradas vecinas para feature PROX. Default: topologia/paradas_vecinas_1500m.csv.
    line_ap_path : Path
        Parquet de LINE a nivel punto_acceso (cercanías). Default: MDLF/LINE/output/line_cercanias_ap.parquet.
    n_topics_cercanias : int
        K para RegionFunctionLDA cercanías.
    test_size : float
        Proporción del conjunto de test. Default 0.2.
    n_weeks : int
        Semanas de historial para crowd features. Default 2.
    n_heads : int
        H — número de cabezas de atención bilineal OD. Default 4 (paper).
    d_t : int
        D_T — dimensión del embedding temporal (tabla conjunta daytype×slot). Default 16.
    cnn_channels : int
        C — canales de las capas convolucionales. Default 16 (paper).
    kernel_size : int
        M — tamaño de kernel Conv1d sobre N estaciones. Default 1 (por estación independiente).
        M=3 añade contexto local entre estaciones vecinas. Usar valores impares.
    lr : float
        Learning rate del optimizador. Default 0.001 (AdamW) o 0.01 (SGD, paper).
    optimizer : str
        'adamw' (default) o 'sgd' (paper usa SGD con momentum 0–0.9).
    batch_size : int
        Tamaño del batch en entrenamiento. Default 64.
    n_epochs : int
        Número de épocas. Default 20.
    random_state : int
    """

    def __init__(
        self,
        nombre_almacen      : str,
        hist_individual_paths: list[Path] | None = None,
        ttp_paths           : list[str] | None = None,
        cubo_hiperparams    : dict | None = None,
        od_matrices_dir     : Path = _OD_MATRICES_DIR,
        od_matrices_subdirs : list[str] | None = None,
        paradas_vecinas_path: Path = _PARADAS_VECINAS,
        line_ap_path        : Path = _LINE_AP,
        n_topics_cercanias  : int  = 16,
        lda_alpha           : float | None = None,
        lda_beta            : float | None = None,
        lda_max_iter        : int  = 20,
        lda_use_periods     : bool  = False,
        lda_min_df          : int  = 1,
        lda_log_counts      : bool  = False,
        lda_token_granularity: str  = "station",
        lda_min_station_trips: int  = 0,
        lda_n_init          : int  = 1,
        lda_filter_prefixes : list | None = None,
        lda_slot_minutes    : int  = 15,
        slot_minutes        : int  = 30,
        test_size           : float = 0.2,
        n_weeks             : int  = 2,
        n_heads             : int  = 4,
        d_t                 : int  = 16,
        cnn_channels        : int  = 16,
        kernel_size         : int  = 1,
        include_prox        : bool  = False,
        td_mode             : str  = "context",
        include_neighborhood_crowd: bool  = False,
        dominance_ratio     : float = 2.0,
        min_days_residence  : int  = 5,
        min_neighbors       : int  = 10,
        lr                  : float = 0.01,
        weight_decay        : float = 0.01,
        optimizer           : str  = "sgd",
        batch_size          : int  = 64,
        n_epochs            : int  = 20,
        lr_scheduler        : bool  = False,
        random_state        : int  = 42,
        ablation            : set[str] | None = None,
        norm                : str | None = "l2",
        split_date          : str | None = None,
        line_meta           : dict | None = None,
        holiday_dates       : set | frozenset = frozenset(),
    ) -> None:
        # ablation: conjunto de módulos a desactivar para estudio de ablación.
        # Valores posibles: "lda", "line", "individual", "crowd", "neighborhood_crowd"
        # Ejemplo: ablation={"lda", "line"} → sin Region Features → H_eff=0
        # Nota: ablation={"crowd"} desactiva también neighborhood_crowd implícitamente.

        # ── Cargar inputs desde almacén ────────────────────────────────────
        _input_dir = _MDLF_ALMACEN / nombre_almacen
        if not _input_dir.exists():
            raise FileNotFoundError(
                f"No se encuentra el almacén '{nombre_almacen}' en {_MDLF_ALMACEN}. "
                f"Ejecuta primero el pipeline de cubo izquierdo con hipótesis 21."
            )
        self.nombre_almacen        = nombre_almacen
        self.x_pred_checked        = pd.read_parquet(_input_dir / "x_pred_checked.parquet")
        self.unknown               = pd.read_parquet(_input_dir / "unknown.parquet")
        self.x_test                = pd.read_parquet(_input_dir / "x_test.parquet")
        self.y_test                = pd.read_parquet(_input_dir / "y_test.parquet")
        _ssp = pd.read_parquet(_input_dir / "subidas_sin_pred.parquet")
        self.subidas_sin_pred      = _ssp if not _ssp.empty else None
        print(f"Inputs MDLF cargados desde {_input_dir}")
        # ──────────────────────────────────────────────────────────────────

        self.hist_individual_paths = [Path(p) for p in hist_individual_paths] if hist_individual_paths else []
        self.ttp_paths            = ttp_paths
        self.cubo_hiperparams     = cubo_hiperparams or {}
        self.od_matrices_dir      = Path(od_matrices_dir)
        self.od_matrices_subdirs  = od_matrices_subdirs
        self.paradas_vecinas_path = Path(paradas_vecinas_path)
        self.line_ap_path         = Path(line_ap_path)
        self.n_topics_cercanias   = n_topics_cercanias
        self.lda_alpha            = lda_alpha
        self.lda_beta             = lda_beta
        self.lda_max_iter         = lda_max_iter
        self.lda_use_periods      = lda_use_periods
        self.lda_min_df           = lda_min_df
        self.lda_log_counts       = lda_log_counts
        self.lda_token_granularity = lda_token_granularity
        self.lda_min_station_trips = lda_min_station_trips
        self.lda_n_init           = lda_n_init
        self.lda_filter_prefixes  = lda_filter_prefixes if lda_filter_prefixes is not None else ["04"]
        self.lda_slot_minutes     = lda_slot_minutes
        self.slot_minutes         = slot_minutes
        self.test_size            = test_size
        self.n_weeks                    = n_weeks
        self.include_neighborhood_crowd = include_neighborhood_crowd
        self.dominance_ratio            = dominance_ratio
        self.min_days_residence         = min_days_residence
        self.min_neighbors              = min_neighbors
        self.n_heads              = n_heads
        self.d_t                  = d_t
        self.include_prox         = include_prox
        self.td_mode              = td_mode
        self.cnn_channels         = cnn_channels
        self.kernel_size          = kernel_size
        self.lr                   = lr
        self.weight_decay         = weight_decay
        self.optimizer_type       = optimizer
        self.batch_size           = batch_size
        self.n_epochs             = n_epochs
        self.lr_scheduler         = lr_scheduler
        self.random_state         = random_state
        self.ablation             = frozenset(ablation) if ablation else frozenset()
        self.norm                 = norm
        self.split_date           = pd.Timestamp(split_date) if split_date else None
        self.line_meta            = line_meta or {}
        self.holiday_dates        = frozenset(holiday_dates)

        # Atributos rellenados a lo largo de run()
        self.labeled_c      : pd.DataFrame | None = None
        self.train_c        : pd.DataFrame | None = None
        self.test_c         : pd.DataFrame | None = None
        self.theta_c        : pd.DataFrame | None = None   # [N × K]  LDA
        self.line_c         : pd.DataFrame | None = None   # [N × 64] LINE
        self.station_index  : list[str]    | None = None
        self.hist_ind_full  : pd.DataFrame | None = None   # historial largo (hist_builder)
        self.X_ind_train    : pd.DataFrame | None = None
        self.X_ind_test     : pd.DataFrame | None = None
        self.X_crowd_train  : pd.DataFrame | None = None
        self.X_crowd_test   : pd.DataFrame | None = None
        self.X_nbcrowd_train: pd.DataFrame | None = None
        self.X_nbcrowd_test : pd.DataFrame | None = None
        # Componentes del tensor V (separados para forward pass en training)
        self.V_static_train : np.ndarray   | None = None   # (n_train, N, D_static)
        self.V_static_test  : np.ndarray   | None = None   # (n_test,  N, D_static)
        self.V_D            : np.ndarray   | None = None   # (N, D_R) region features destinos
        self.v_o_train      : np.ndarray   | None = None   # (n_train, D_R) origen por viaje
        self.v_o_test       : np.ndarray   | None = None   # (n_test,  D_R)
        self.temporal_train : np.ndarray   | None = None   # (n_train,) int — índice embedding
        self.temporal_test  : np.ndarray   | None = None   # (n_test,)  int
        self.y_train        : pd.Series    | None = None
        self.y_mdlf_test    : pd.Series    | None = None
        self.model          : _DestinationCNN | None = None
        self.train_losses   : list[float]  | None = None   # loss medio por época
        self.test_accuracy  : float        | None = None   # top-1
        self.test_accuracy5 : float        | None = None   # top-5
        self._y_true_eval   : np.ndarray   | None = None   # índices verdaderos (test)
        self._y_pred_eval   : np.ndarray   | None = None   # índices predichos top-1 (test)
        self.origin_idx_train: np.ndarray  | None = None   # (n_train,) idx origen en station_index
        self.origin_idx_test : np.ndarray  | None = None   # (n_test,)  idx origen en station_index

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self, run_name: str | None = None, descripcion: str | None = None) -> "PipelineMDLF":
        """Ejecuta el pipeline MDLF completo.

        Parameters
        ----------
        run_name : str | None
            Prefijo para el fichero de log. Si se proporciona, el log se guarda
            como ``runs/<run_name>_<timestamp>.log``. Si es None, se usa
            ``runs/run_<timestamp>.log``.
        descripcion : str | None
            Texto explicativo del experimento. Se muestra en las primeras líneas
            del log, antes de los hiperparámetros.
        """
        import time as _time
        _ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        _prefix  = run_name if run_name else "run"
        _logpath = _RUNS_DIR / f"{_prefix}_{_ts}.log"

        with _Tee(_logpath):
            _t0_total = _time.time()
            print("\n" + "="*60)
            print(f"  PIPELINE MDLF — inicio  [{_ts}]")
            if run_name:
                print(f"  Nombre        : {run_name}")
            print(f"  Log: {_logpath}")
            if descripcion:
                print("  " + "─"*58)
                for linea in descripcion.strip().splitlines():
                    print(f"  {linea}")
            print("="*60)

            # ── Hiperparámetros ────────────────────────────────────────────
            print("\n  ── Hiperparámetros ──────────────────────────────────────")
            if self.line_meta:
                _lm = self.line_meta
                print(f"  LINE        : dim={_lm.get('dim','?')} | asymmetric={_lm.get('asymmetric','?')} | num_negative={_lm.get('num_negative','?')} | lr={_lm.get('lr','?')} | fo_epochs={_lm.get('num_epochs_fo','?')} | so_epochs={_lm.get('num_epochs_so','?')}")
            print(f"  LDA         : K={self.n_topics_cercanias} | alpha={self.lda_alpha} | beta={self.lda_beta} | max_iter={self.lda_max_iter}")
            print(f"  LDA extras  : use_periods={self.lda_use_periods} | min_df={self.lda_min_df} | log_counts={self.lda_log_counts} | granularity={self.lda_token_granularity} | min_trips={self.lda_min_station_trips}")
            print(f"  CNN         : C={self.cnn_channels} | M={self.kernel_size} | H={self.n_heads} | D_T={self.d_t}")
            print(f"  Train       : epochs={self.n_epochs} | batch={self.batch_size} | optimizer={self.optimizer_type} | lr={self.lr} | wd={self.weight_decay} | scheduler={self.lr_scheduler}")
            print(f"  Features    : n_weeks={self.n_weeks} | include_prox={self.include_prox} | td_mode={self.td_mode!r} | include_neighborhood_crowd={self.include_neighborhood_crowd} | norm={self.norm}")
            print(f"  Split       : split_date={self.split_date.date() if self.split_date else None} | test_size={self.test_size} | random_state={self.random_state} | slot_minutes={self.slot_minutes}")
            if self.ablation:
                print(f"  Ablación    : {sorted(self.ablation)}")
            print("  " + "─"*58)
            logger.info(
                "Hiperparámetros — K=%d alpha=%s beta=%s max_iter=%d "
                "use_periods=%s min_df=%d log_counts=%s granularity=%s min_trips=%d | "
                "C=%d M=%d H=%d D_T=%d | "
                "epochs=%d batch=%d opt=%s lr=%g wd=%g scheduler=%s | "
                "n_weeks=%d prox=%s td_mode=%s nbcrowd=%s norm=%s | "
                "split_date=%s test_size=%g seed=%d slot_min=%d | ablation=%s",
                self.n_topics_cercanias, self.lda_alpha, self.lda_beta, self.lda_max_iter,
                self.lda_use_periods, self.lda_min_df, self.lda_log_counts,
                self.lda_token_granularity, self.lda_min_station_trips,
                self.cnn_channels, self.kernel_size, self.n_heads, self.d_t,
                self.n_epochs, self.batch_size, self.optimizer_type, self.lr, self.weight_decay, self.lr_scheduler,
                self.n_weeks, self.include_prox, self.td_mode, self.include_neighborhood_crowd, self.norm,
                self.split_date.date() if self.split_date else None,
                self.test_size, self.random_state, self.slot_minutes,
                sorted(self.ablation) if self.ablation else "none",
            )

            _t = _time.time()
            print("\n[1/9] Construyendo conjunto supervisado...")
            self._build_labeled_set()
            print(f"      OK ({_time.time()-_t:.1f}s)")

            _t = _time.time()
            print("[2/9] Split train/test...")
            self._split()
            print(f"      OK ({_time.time()-_t:.1f}s)")

            if self.ablation:
                print(f"      [Ablación activa: módulos desactivados = {sorted(self.ablation)}]")

            _t = _time.time()
            print("[3/9] Ajustando LDA (Region Features)...")
            if "lda" not in self.ablation:
                self._fit_lda()
            else:
                self._skip_lda()
                print("      SKIP (ablación)")
            print(f"      OK ({_time.time()-_t:.1f}s)")

            _t = _time.time()
            print("[4/9] Cargando embeddings LINE...")
            if "line" not in self.ablation:
                self._load_line()
            else:
                self._skip_line()
                print("      SKIP (ablación)")
            print(f"      OK ({_time.time()-_t:.1f}s)")

            _t = _time.time()
            print("[5/9] Extrayendo Individual Features...")
            if "individual" not in self.ablation:
                self._extract_individual_features()
            else:
                self._skip_individual()
                print("      SKIP (ablación)")
            print(f"      OK ({_time.time()-_t:.1f}s)")

            _t = _time.time()
            print("[6/9] Extrayendo Crowd Features...")
            if "crowd" not in self.ablation:
                self._extract_crowd_features()
            else:
                self._skip_crowd()
                print("      SKIP (ablación)")
            print(f"      OK ({_time.time()-_t:.1f}s)")

            _t = _time.time()
            print("[7/9] Construyendo tensores V...")
            self._build_V()
            print(f"      OK ({_time.time()-_t:.1f}s)")

            _t = _time.time()
            print("[8/9] Inicializando CNN...")
            self._init_model()
            print(f"      OK ({_time.time()-_t:.1f}s)")

            _t = _time.time()
            print("[9/9] Entrenando modelo...")
            self._train_model()
            print(f"      OK ({_time.time()-_t:.1f}s)")

            print("\n[Eval] Evaluando en test...")
            self._evaluate()

            print("\n[Attr] Atribución por grupo de features...")
            self.attribution_by_group()

            print("\n[Pred] Análisis de predicciones...")
            self.prediction_analysis(top_n=10, log=True)

            _elapsed = _time.time() - _t0_total
            print("\n" + "="*60)
            print(f"  PIPELINE MDLF — completado en {_elapsed:.1f}s")
            print("="*60 + "\n")

        self.log_path = _logpath
        return self

    # ------------------------------------------------------------------
    # Paso 1 — Conjunto supervisado
    # ------------------------------------------------------------------

    def _build_labeled_set(self) -> None:
        """
        Intersecta los viajes de cercanías sin predicción de descenso (self.unknown)
        con los pares de crear_mapeos (x_test/y_test) para obtener los viajes
        con etiqueta real de descenso.
        """
        unlinked_c = self.unknown[
            self.unknown["tipo_transporte"] == "cercanias"
        ].copy()

        if self.subidas_sin_pred is not None:
            sin_pred_c = self.subidas_sin_pred[
                self.subidas_sin_pred["tipo_transporte"] == "cercanias"
            ].copy()
            unlinked_c = pd.concat(
                [unlinked_c, sin_pred_c], ignore_index=True
            ).drop_duplicates(subset=["id_tarjeta", "timestamp_validacion"])

        keys = ["id_tarjeta", "timestamp_validacion"]

        # Normalizar timestamps a tz-naive para evitar incompatibilidad con x_test
        # (self.unknown viene de Spark toPandas → tz-aware UTC; x_test es pandas puro)
        unlinked_c["timestamp_validacion"] = pd.to_datetime(
            unlinked_c["timestamp_validacion"]
        ).dt.tz_localize(None)

        # x_test e y_test están alineados por posición: fila i de x_test
        # corresponde a fila i de y_test. Asignamos destino_real directamente
        # antes de mergear, en lugar de hacer un segundo merge por timestamp
        # (y_test tiene timestamps de bajada, no de subida → no coincidirían).
        x_test_norm = self.x_test.copy()
        x_test_norm["timestamp_validacion"] = pd.to_datetime(
            x_test_norm["timestamp_validacion"]
        ).dt.tz_localize(None)
        x_test_norm["destino_real"] = self.y_test["punto_acceso"].values

        labeled = unlinked_c.merge(
            x_test_norm[keys + ["destino_real"]].drop_duplicates(subset=keys),
            on=keys,
            how="inner",
        )
        labeled = labeled[labeled["destino_real"].notna()].reset_index(drop=True)

        # Eliminar self-trips: viajes donde el origen == destino real.
        # Son errores de datos (y_test asigna la misma estación como bajada y subida).
        # Si no se eliminan, el enmascaramiento de origen en la loss produce loss=inf.
        n_before = len(labeled)
        labeled = labeled[labeled["punto_acceso"] != labeled["destino_real"]].reset_index(drop=True)
        n_self = n_before - len(labeled)
        if n_self:
            logger.warning("Eliminados %d self-trips del labeled set (punto_acceso == destino_real)", n_self)

        self.labeled_c = labeled
        logger.info(
            "Labeled set cercanías UNLINKED: %d samples "
            "(de %d UNLINKED totales, %d en x_test)",
            len(labeled), len(unlinked_c), len(self.x_test),
        )
        print(f"      Cercanías sin predicción (unknown{' + sin_pred' if self.subidas_sin_pred is not None else ''}) : {len(unlinked_c)}")
        print(f"      Con bajada real en x_test  : {len(labeled)}")
        if n_self:
            print(f"      Self-trips eliminados      : {n_self}")
        print(f"      Destinos únicos            : {labeled['destino_real'].nunique()}")
        if len(labeled) == 0:
            raise ValueError(
                "El labeled set está vacío. Comprueba que x_pred_checked y "
                "x_test provienen del mismo período."
            )

    # ------------------------------------------------------------------
    # Paso 2 — Split
    # ------------------------------------------------------------------

    def _split(self) -> None:
        """Split estratificado por estación de destino real, o temporal si split_date."""
        if self.split_date is not None:
            _raw = pd.to_datetime(self.labeled_c["timestamp_validacion"])
            ts = _raw.dt.tz_convert(None) if _raw.dt.tz is not None else _raw
            mask_train = ts < self.split_date
            train = self.labeled_c[mask_train]
            test  = self.labeled_c[~mask_train]
            if len(train) == 0 or len(test) == 0:
                raise ValueError(
                    f"Split temporal con split_date={self.split_date} produce "
                    f"train={len(train)} / test={len(test)}. Revisa la fecha."
                )
            print(f"      Split TEMPORAL ({self.split_date.date()}): "
                  f"train={len(train)} | test={len(test)}")
        else:
            try:
                train, test = train_test_split(
                    self.labeled_c,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=self.labeled_c["destino_real"],
                )
            except ValueError:
                # Si alguna clase tiene <2 muestras, stratify falla → sin stratify
                logger.warning(
                    "Stratify falló (clases con <2 muestras). Usando split aleatorio."
                )
                train, test = train_test_split(
                    self.labeled_c,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

        self.train_c      = train.reset_index(drop=True)
        self.test_c       = test.reset_index(drop=True)
        self.y_train      = self.train_c["destino_real"].reset_index(drop=True)
        self.y_mdlf_test  = self.test_c["destino_real"].reset_index(drop=True)
        logger.info(
            "Split: train=%d | test=%d | destinos únicos train=%d",
            len(self.train_c), len(self.test_c),
            self.y_train.nunique(),
        )
        print(f"      train={len(self.train_c)} | test={len(self.test_c)} | "
              f"destinos únicos={self.y_train.nunique()}")

    # ------------------------------------------------------------------
    # Paso 3 — LDA
    # ------------------------------------------------------------------

    def _fit_lda(self) -> None:
        """
        Carga matrices OD y ajusta RegionFunctionLDA para cercanías.
        Produce self.theta_c [N × K] y self.station_index.

        Si od_matrices_subdirs está definido, solo carga las subcarpetas
        indicadas (nombres relativos a od_matrices_dir). En caso contrario
        carga recursivamente todo lo que haya bajo od_matrices_dir.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from LDA.run_lda import load_od_matrices
        from LDA.region_function_lda import RegionFunctionLDA

        if self.od_matrices_subdirs is not None:
            # Cargar solo las subcarpetas especificadas y concatenar
            dfs = []
            for subdir in self.od_matrices_subdirs:
                subpath = self.od_matrices_dir / subdir
                if not subpath.exists():
                    raise FileNotFoundError(
                        f"Subcarpeta de matrices OD no encontrada: {subpath}"
                    )
                dfs.append(load_od_matrices(subpath))
            import pandas as _pd
            od_df = _pd.concat(dfs, ignore_index=True)
            logger.info(
                "LDA: cargadas %d subcarpetas → %d filas totales",
                len(self.od_matrices_subdirs), len(od_df),
            )
        else:
            od_df = load_od_matrices(self.od_matrices_dir)

        model = RegionFunctionLDA(
            n_topics=self.n_topics_cercanias,
            slot_minutes=self.lda_slot_minutes,
            filter_prefixes=self.lda_filter_prefixes,
            alpha=self.lda_alpha,
            beta=self.lda_beta,
            max_iter=self.lda_max_iter,
            use_periods=self.lda_use_periods,
            min_df=self.lda_min_df,
            log_counts=self.lda_log_counts,
            token_granularity=self.lda_token_granularity,
            min_station_trips=self.lda_min_station_trips,
            n_init=self.lda_n_init,
            random_state=self.random_state,
            norm=self.norm,
            holiday_dates=self.holiday_dates,
        )
        self.theta_c = model.fit_transform(od_df)        # [N × K]
        self.station_index = self.theta_c.index.tolist()
        logger.info(
            "LDA cercanías: %d estaciones × %d tópicos",
            *self.theta_c.shape,
        )
        print(f"      theta_c: {self.theta_c.shape}  (N estaciones × K tópicos)")

    def _skip_lda(self) -> None:
        """Ablación: sin LDA → theta_c de ancho 0, station_index desde LINE o vacío."""
        # station_index se necesita para alinear LINE y features; lo fijamos
        # a las estaciones que tenga LINE (se resuelve en _skip_line si también falta).
        # Por ahora dejamos station_index=None; _load_line lo rellenará si está activo.
        self.theta_c       = None   # se tratará como array (N, 0) en _build_V
        self.station_index = None   # se rellena por LINE o por _build_V con fallback
        logger.info("Ablación LDA: theta_c desactivado")
        print("      theta_c: desactivado (ablación)")

    # ------------------------------------------------------------------
    # Paso 4 — LINE
    # ------------------------------------------------------------------

    def _load_line(self) -> None:
        """
        Carga los embeddings LINE de cercanías a nivel punto_acceso.
        Produce self.line_c [N × 64] alineado con self.station_index.
        """
        if not self.line_ap_path.exists():
            raise FileNotFoundError(
                f"No se encuentra line_cercanias_ap.parquet en: {self.line_ap_path}\n"
                "Ejecuta primero MDLF/LINE/validacion_line.ipynb."
            )
        line_raw = pd.read_parquet(self.line_ap_path)   # índice = punto_acceso

        # Si station_index es None (LDA ablado), se toma directamente del parquet LINE
        if self.station_index is None:
            self.station_index = list(line_raw.index)
            logger.info("station_index derivado de LINE (%d estaciones)", len(self.station_index))

        # Alinear con station_index (estaciones del LDA)
        missing = [s for s in self.station_index if s not in line_raw.index]
        if missing:
            logger.warning(
                "%d estaciones de LDA sin vector LINE — se rellenan con ceros. "
                "Ejemplo: %s",
                len(missing), missing[:3],
            )
        self.line_c = line_raw.reindex(self.station_index).fillna(0.0)

        if self.norm:
            from sklearn.preprocessing import normalize as _normalize
            self.line_c = pd.DataFrame(
                _normalize(self.line_c.values, norm=self.norm, axis=1),
                index=self.line_c.index,
                columns=self.line_c.columns,
            )

        logger.info(
            "LINE cercanías: %d estaciones × %d dims (norm=%s)",
            *self.line_c.shape, self.norm,
        )
        n_missing = sum(s not in line_raw.index for s in self.station_index)
        print(f"      line_c: {self.line_c.shape}  ({n_missing} estaciones rellenas con ceros)  norm={self.norm}")

    def _skip_line(self) -> None:
        """Ablación: sin LINE → line_c de ancho 0. Si LDA también está off, station_index desde labeled_c."""
        if self.station_index is None:
            # Ni LDA ni LINE activos: derivar station_index de las estaciones del conjunto etiquetado
            self.station_index = sorted(self.labeled_c["punto_acceso"].dropna().unique().tolist())
            logger.info(
                "station_index derivado de labeled_c (%d estaciones) — LDA y LINE ambos ablados",
                len(self.station_index),
            )
            print(f"      station_index: {len(self.station_index)} estaciones (fallback desde labeled_c)")
        self.line_c = pd.DataFrame(
            np.zeros((len(self.station_index), 0), dtype=np.float32),
            index=self.station_index,
        )
        logger.info("Ablación LINE: line_c desactivado")
        print("      line_c: desactivado (ablación)")

    # ------------------------------------------------------------------
    # Paso 5 — Extracción de features individuales y crowd
    # ------------------------------------------------------------------

    def _extract_features(self) -> None:
        """Orquesta la extracción de Individual y Crowd features."""
        self._extract_individual_features()
        self._extract_crowd_features()

    def _extract_individual_features(self) -> None:
        """
        IndividualFeatureExtractor entrenado SOLO sobre usuarios en train ∪ test.

        Historial (prioridad):
          1. x_pred_checked_hist_indiv_mdlf.parquet en caché del almacén
          2. Construir desde TTP CSVs si ttp_paths y cubo_hiperparams están definidos
          3. Fallback: x_pred_checked del almacén (solo el período actual)

        Internamente separa:
          · cercanías           → TD, SD, TSD, TSI
          · metro/bus/EMT/otros → PROX
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from INDIVIDUAL.individual_features import IndividualFeatureExtractor
        from INDIVIDUAL.hist_builder import IndividualHistBuilder

        users = set(self.train_c["id_tarjeta"]) | set(self.test_c["id_tarjeta"])

        builder = IndividualHistBuilder(
            nombre_almacen=self.nombre_almacen,
            ttp_paths=self.ttp_paths,
            hiperparams=self.cubo_hiperparams,
        )

        if builder.is_cached():
            hist_ind_full = builder.load()
            print(f"      Historial individual cargado desde caché del almacén "
                  f"({len(hist_ind_full):,} filas).")
        elif self.ttp_paths:
            print("      Construyendo historial individual desde TTP CSVs...")
            hist_ind_full = builder.build()
        else:
            import warnings
            warnings.warn(
                "No hay historial individual en caché ni ttp_paths definidos. "
                "Usando x_pred_checked como fallback (solo período actual).",
                stacklevel=2,
            )
            hist_ind_full = self.x_pred_checked

        # Guardamos para reutilizar en _extract_neighborhood_crowd_features()
        self.hist_ind_full = hist_ind_full

        hist_ind = hist_ind_full[hist_ind_full["id_tarjeta"].isin(users)]
        print(f"      Historial filtrado a {len(users):,} usuarios: {len(hist_ind):,} filas")

        ext = IndividualFeatureExtractor(
            red="04",
            station_index=self.station_index,
            include_prox=self.include_prox,
            td_mode=self.td_mode,
            paradas_vecinas_path=self.paradas_vecinas_path,
            slot_minutes=self.slot_minutes,
            norm=self.norm,
            holiday_dates=self.holiday_dates,
        )
        ext.fit(hist_ind)
        self.X_ind_train = ext.transform(self.train_c)
        self.X_ind_test  = ext.transform(self.test_c)
        logger.info(
            "Individual features — train: %s | test: %s",
            self.X_ind_train.shape, self.X_ind_test.shape,
        )
        print(f"      X_ind_train: {self.X_ind_train.shape}  "
              f"(n_viajes × N·{6 if not self.include_prox else 8}+1)")
        print(f"      Usuarios en historial: {len(users)}")

    def _extract_crowd_features(self) -> None:
        """
        CrowdFeatureExtractor entrenado sobre TODOS los usuarios (solo cercanías).
        """
        spec = importlib.util.spec_from_file_location("crowd_features", _CROWD_FILE)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        CrowdFeatureExtractor = mod.CrowdFeatureExtractor

        ext = CrowdFeatureExtractor(
            red="04",
            station_index=self.station_index,
            n_weeks=self.n_weeks,
            slot_minutes=self.slot_minutes,
            norm=self.norm,
            holiday_dates=self.holiday_dates,
        )
        # Filtrar historial crowd al período de train para evitar leakage:
        # x_pred_checked contiene viajes BAJADA/TRANSBORDO de todo el período
        # (incluyendo test). Sin filtro, la ventana semanal de viajes de test
        # tardíos (ej. 28-31 mar) incluiría datos del período de test.
        if self.split_date is not None:
            self._hist_crowd = self.x_pred_checked[
                pd.to_datetime(self.x_pred_checked["timestamp_validacion"]).dt.tz_localize(None)
                < self.split_date
            ]
            print(f"      Historial crowd filtrado a < {self.split_date.date()}: "
                  f"{len(self._hist_crowd):,} filas (de {len(self.x_pred_checked):,})")
        else:
            self._hist_crowd = self.x_pred_checked
        ext.fit(self._hist_crowd)
        _ref = self.split_date.date() if self.split_date is not None else None
        self.X_crowd_train = ext.transform(self.train_c)
        self.X_crowd_test  = ext.transform(self.test_c, reference_date=_ref)
        logger.info(
            "Crowd features — train: %s | test: %s",
            self.X_crowd_train.shape, self.X_crowd_test.shape,
        )
        print(f"      X_crowd_train: {self.X_crowd_train.shape}  "
              f"(n_viajes × N·{self.n_weeks}+1)")
        has_signal = self.X_crowd_train["has_crowd_signal"].mean()
        print(f"      Viajes con señal crowd: {has_signal:.1%}")

        # ── Neighborhood Crowd (opcional) ─────────────────────────────
        if self.include_neighborhood_crowd and "neighborhood_crowd" not in self.ablation:
            self._extract_neighborhood_crowd_features()
        else:
            self._skip_neighborhood_crowd()

    def _extract_neighborhood_crowd_features(self) -> None:
        """
        Infiere paradas de residencia y extrae crowd features del vecindario.
        Usa el mismo historial que IndividualFeatureExtractor.
        """
        import sys as _sys
        spec_r = importlib.util.spec_from_file_location("residence_inferrer", _RESIDENCE_FILE)
        mod_r  = importlib.util.module_from_spec(spec_r)
        _sys.modules["residence_inferrer"] = mod_r
        spec_r.loader.exec_module(mod_r)

        spec_n = importlib.util.spec_from_file_location("neighborhood_crowd", _NBCROWD_FILE)
        mod_n  = importlib.util.module_from_spec(spec_n)
        spec_n.loader.exec_module(mod_n)

        ResidenceInferrer        = mod_r.ResidenceInferrer
        NeighborhoodCrowdExtractor = mod_n.NeighborhoodCrowdExtractor

        # Mismo historial largo que individual features (ya cargado en _extract_individual_features)
        inferrer = ResidenceInferrer(
            dominance_ratio=self.dominance_ratio,
            min_days=self.min_days_residence,
        )
        inferrer.fit(self.hist_ind_full)

        cov = inferrer.coverage()
        print(f"      ResidenceInferrer: {cov['n_confirmed']}/{cov['n_users_total']} "
              f"usuarios con residencia confirmada ({cov['pct_confirmed']:.1%})")
        print(f"      Razones: {cov['reason_counts']}")

        ext = NeighborhoodCrowdExtractor(
            red="04",
            station_index=self.station_index,
            n_weeks=self.n_weeks,
            min_neighbors=self.min_neighbors,
            slot_minutes=self.slot_minutes,
            norm=self.norm,
            holiday_dates=self.holiday_dates,
        )
        # Reutiliza el historial ya filtrado por _extract_crowd_features()
        ext.fit(self._hist_crowd, inferrer)

        _ref = self.split_date.date() if self.split_date is not None else None
        self.X_nbcrowd_train = ext.transform(self.train_c)
        self.X_nbcrowd_test  = ext.transform(self.test_c, reference_date=_ref)

        has_signal = self.X_nbcrowd_train["has_nbcrowd_signal"].mean()
        print(f"      X_nbcrowd_train: {self.X_nbcrowd_train.shape}  "
              f"(viajes con señal: {has_signal:.1%})")
        logger.info(
            "Neighborhood crowd — train: %s | test: %s | cobertura residencia: %.1f%%",
            self.X_nbcrowd_train.shape, self.X_nbcrowd_test.shape,
            cov["pct_confirmed"] * 100,
        )

    def _skip_neighborhood_crowd(self) -> None:
        """Desactiva neighborhood crowd → X_nbcrowd de ancho 0."""
        n_tr = len(self.train_c) if self.train_c is not None else 0
        n_te = len(self.test_c)  if self.test_c  is not None else 0
        self.X_nbcrowd_train = pd.DataFrame(
            np.zeros((n_tr, 1), dtype=np.float32), columns=["has_nbcrowd_signal"]
        )
        self.X_nbcrowd_test = pd.DataFrame(
            np.zeros((n_te, 1), dtype=np.float32), columns=["has_nbcrowd_signal"]
        )
        self._D_nbcrowd_eff = 0

    def _skip_individual(self) -> None:
        """Ablación: sin Individual Features → matrices de ancho 0.
        Carga igualmente self.hist_ind_full desde caché porque
        _extract_neighborhood_crowd_features() lo necesita para ResidenceInferrer.
        """
        N = len(self.station_index)
        n_tr, n_te = len(self.train_c), len(self.test_c)
        self.X_ind_train = pd.DataFrame(np.zeros((n_tr, N * 0 + 1), dtype=np.float32))
        self.X_ind_test  = pd.DataFrame(np.zeros((n_te, N * 0 + 1), dtype=np.float32))
        self._D_ind_eff = 0
        logger.info("Ablación Individual: X_ind desactivado")
        print("      X_ind: desactivado (ablación)")
        # Cargar historial para que ResidenceInferrer (neighborhood crowd) pueda usarlo
        if self.include_neighborhood_crowd and "neighborhood_crowd" not in self.ablation:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from INDIVIDUAL.hist_builder import IndividualHistBuilder
            builder = IndividualHistBuilder(
                nombre_almacen=self.nombre_almacen,
                ttp_paths=self.ttp_paths,
                hiperparams=self.cubo_hiperparams,
            )
            if builder.is_cached():
                self.hist_ind_full = builder.load()
                print(f"      hist_ind_full cargado desde caché ({len(self.hist_ind_full):,} filas) "
                      f"— necesario para neighborhood crowd")
            else:
                logger.warning("hist_ind_full no disponible en caché — neighborhood crowd desactivado")
                self._skip_neighborhood_crowd()
                self.include_neighborhood_crowd = False

    def _skip_crowd(self) -> None:
        """Ablación: sin Crowd Features → matrices de ancho 0.
        También inicializa neighborhood crowd a cero, ya que NeighborhoodCrowdExtractor
        depende de self._hist_crowd que solo existe si _extract_crowd_features() se ejecuta.
        """
        N = len(self.station_index)
        n_tr, n_te = len(self.train_c), len(self.test_c)
        self.X_crowd_train = pd.DataFrame(np.zeros((n_tr, N * 0 + 1), dtype=np.float32))
        self.X_crowd_test  = pd.DataFrame(np.zeros((n_te, N * 0 + 1), dtype=np.float32))
        self._D_crowd_eff = 0
        logger.info("Ablación Crowd: X_crowd desactivado")
        print("      X_crowd: desactivado (ablación)")
        # Neighborhood crowd depende del historial crowd → también se desactiva
        self._skip_neighborhood_crowd()
        print("      X_nbcrowd: desactivado (dependencia de crowd ablada)")

    # ------------------------------------------------------------------
    # Paso 6 — Construcción del tensor V ∈ R^(n_trips, N, D)
    # ------------------------------------------------------------------

    def _build_V(self) -> None:
        """
        Prepara todos los tensores necesarios para el forward pass de _DestinationCNN.

        Almacena por separado:
          · V_static  = [F_u ‖ F_crowd] ∈ R^(n, N, D_static)   D_static = 8 + n_weeks
          · V_D       = region features de todos los destinos ∈ R^(N, D_R)
          · v_o       = region features de la estación origen por viaje ∈ R^(n, D_R)
          · temporal  = índice de embedding temporal por viaje ∈ Z^(n,)

        F'_R ∈ R^(n, N, H) se calcula en el forward de _DestinationCNN para que
        los gradientes fluyan a través de los pesos de _ODRegionModule (E_table, W1, W2).
        """
        N = len(self.station_index)

        # ── Region Features V_D ∈ R^(N, D_R) ─────────────────────────────
        parts_vd = []
        if self.theta_c is not None:
            parts_vd.append(self.theta_c.values)
        if self.line_c is not None and self.line_c.shape[1] > 0:
            parts_vd.append(self.line_c.values)

        if parts_vd:
            self.V_D = np.concatenate(parts_vd, axis=1).astype(np.float32)
        else:
            self.V_D = np.zeros((N, 0), dtype=np.float32)

        D_R = self.V_D.shape[1]
        # Si no hay Region Features, o si se abla el módulo temporal, se desactiva la OD correlation
        if "temporal" in self.ablation:
            self.n_heads_eff = 0
            logger.info("Ablación temporal: OD correlation module desactivado (n_heads_eff=0)")
            print("      OD correlation module: desactivado (ablación temporal)")
        else:
            self.n_heads_eff = self.n_heads if D_R > 0 else 0

        # ── Temporal index y v_o ──────────────────────────────────────────
        station_to_idx = {s: i for i, s in enumerate(self.station_index)}

        def _extract_od_inputs(df: pd.DataFrame, split: str):
            _od_util = ODRegionCorrelation(D_R=max(D_R, 1), D_T=self.d_t, n_heads=1, slot_minutes=self.slot_minutes, holiday_dates=self.holiday_dates)
            temporal_idx = _od_util.temporal_index(df)
            if D_R > 0:
                origin_row = df["punto_acceso"].map(station_to_idx)
                n_unk = origin_row.isna().sum()
                if n_unk:
                    logger.warning(
                        "%s: %d viajes con origen fuera de station_index → v_o = cero",
                        split, n_unk,
                    )
                v_o = self.V_D[origin_row.fillna(0).astype(int).values]
            else:
                v_o = np.zeros((len(df), 0), dtype=np.float32)
            return v_o, temporal_idx

        v_o_tr, tidx_tr = _extract_od_inputs(self.train_c, "train")
        v_o_te, tidx_te = _extract_od_inputs(self.test_c,  "test")

        self.v_o_train      = v_o_tr
        self.v_o_test       = v_o_te
        self.temporal_train = tidx_tr
        self.temporal_test  = tidx_te

        # Índice de estación de origen por viaje — para enmascarar self-trips en logits
        # (-1 cuando la estación de subida no está en station_index)
        self.origin_idx_train = (
            self.train_c["punto_acceso"].map(station_to_idx).fillna(-1).astype(int).values
        )
        self.origin_idx_test = (
            self.test_c["punto_acceso"].map(station_to_idx).fillna(-1).astype(int).values
        )

        # ── D_ind, D_crowd y D_nbcrowd efectivos (0 si ablación activa) ────
        D_ind     = getattr(self, "_D_ind_eff",     8 if self.include_prox else 6)
        D_crowd   = getattr(self, "_D_crowd_eff",   self.n_weeks)
        D_nbcrowd = getattr(self, "_D_nbcrowd_eff",
                            self.n_weeks if self.include_neighborhood_crowd else 0)

        # V_static: [F_u ‖ F_crowd ‖ F_nbcrowd] — no incluye F'_R
        self.V_static_train = self._assemble_V_static(
            self.X_ind_train, self.X_crowd_train, self.X_nbcrowd_train,
            N, D_ind, D_crowd, D_nbcrowd,
        )
        self.V_static_test = self._assemble_V_static(
            self.X_ind_test, self.X_crowd_test, self.X_nbcrowd_test,
            N, D_ind, D_crowd, D_nbcrowd,
        )
        D_static = self.V_static_train.shape[2]
        logger.info(
            "Tensores V preparados — train: %s | test: %s | "
            "D_static=%d (D_U=%d + D_W=%d + D_NB=%d) | D_R=%d | H_eff=%d",
            self.V_static_train.shape, self.V_static_test.shape,
            D_static, D_ind, D_crowd, D_nbcrowd, D_R, self.n_heads_eff,
        )
        print(f"      V_static_train : {self.V_static_train.shape}  "
              f"(n_train × N × D_static={D_static}  "
              f"= D_ind={D_ind} + D_crowd={D_crowd} + D_nb={D_nbcrowd})")
        K = self.theta_c.shape[1] if self.theta_c is not None else 0
        L = self.line_c.shape[1]  if (self.line_c is not None and self.line_c.shape[1] > 0) else 0
        print(f"      V_D            : {self.V_D.shape}  "
              f"(N × D_R={D_R}  =  K={K} + L={L})")
        print(f"      v_o_train      : {self.v_o_train.shape}")
        print(f"      temporal_train : {self.temporal_train.shape}  "
              f"idx∈[{self.temporal_train.min()}, {self.temporal_train.max()}]")

    @staticmethod
    def _assemble_V_static(
        X_ind    : pd.DataFrame,
        X_crowd  : pd.DataFrame,
        X_nbcrowd: pd.DataFrame,
        N        : int,
        D_ind    : int,
        D_crowd  : int,
        D_nbcrowd: int,
    ) -> np.ndarray:
        """
        Ensambla V_static = [F_u ‖ F_crowd ‖ F_nbcrowd] ∈ R^(n, N, D_static).

        Descarta la última columna de cada DataFrame (flag has_* signal).
        D_nbcrowd=0 cuando include_neighborhood_crowd=False → bloque omitido.
        """
        n = len(X_ind)
        parts = []
        if D_ind > 0:
            parts.append(X_ind.iloc[:, :-1].values.reshape(n, N, D_ind).astype(np.float32))
        if D_crowd > 0:
            parts.append(X_crowd.iloc[:, :-1].values.reshape(n, N, D_crowd).astype(np.float32))
        if D_nbcrowd > 0:
            parts.append(X_nbcrowd.iloc[:, :-1].values.reshape(n, N, D_nbcrowd).astype(np.float32))
        if parts:
            return np.concatenate(parts, axis=2)               # (n, N, D_static)
        return np.zeros((n, N, 0), dtype=np.float32)

    # ------------------------------------------------------------------
    # Paso 7 — Inicialización del modelo
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Instancia _DestinationCNN con las dimensiones derivadas de los datos."""
        N        = len(self.station_index)
        D_static = self.V_static_train.shape[2]   # D_ind + n_weeks
        D_R      = self.V_D.shape[1]              # K + 64
        n_entries = ODRegionCorrelation.N_DAYTYPES * (1440 // self.slot_minutes)

        torch.manual_seed(self.random_state)
        self.model = _DestinationCNN(
            N         = N,
            D_static  = D_static,
            D_R       = D_R,
            D_T       = self.d_t,
            n_heads   = self.n_heads_eff,   # 0 si LDA+LINE desactivados
            n_entries = n_entries,
            C         = self.cnn_channels,
            M         = self.kernel_size,
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "CNN inicializada — N=%d | D_static=%d | D_R=%d | C=%d | M=%d | "
            "H=%d | D_T=%d | parámetros totales: %d",
            N, D_static, D_R, self.cnn_channels, self.kernel_size,
            self.n_heads, self.d_t, n_params,
        )
        print(f"      N={N} | D_static={D_static} | D_R={D_R} | D={D_static+self.n_heads}")
        print(f"      H={self.n_heads} | D_T={self.d_t} | C={self.cnn_channels} | M={self.kernel_size}")
        print(f"      Parámetros entrenables: {n_params:,}")

    # ------------------------------------------------------------------
    # Paso 8 — Entrenamiento por épocas y batches
    # ------------------------------------------------------------------

    def _train_model(self) -> None:
        """
        Bucle de entrenamiento PyTorch: épocas × batches.

        Optimizador: AdamW (default, lr=0.001) o SGD con momentum 0.9 (paper, lr=0.01).
        Loss: CrossEntropyLoss — incluye SoftMax implícito sobre los N destinos.

        Por época:
          1. Permutación aleatoria de índices de train
          2. Itera en batches de tamaño self.batch_size
          3. forward → logits (batch, N) → loss → backward → optimizer.step
          4. Registra loss medio en self.train_losses
        """
        self.train_losses = []
        n_train = len(self.V_static_train)

        # Labels: índice de la estación destino real dentro de station_index
        station_to_idx = {s: i for i, s in enumerate(self.station_index)}
        y_idx = self.y_train.map(station_to_idx).fillna(-1).astype(int).values

        # Dispositivo: GPU si está disponible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        if torch.cuda.is_available():
            logger.info("Entrenamiento en GPU: %s", torch.cuda.get_device_name(0))
            print(f"      Dispositivo: GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("      Dispositivo: CPU")
        self._device = device  # guardado para _evaluate

        # Tensores fijos en device
        Vs_tr  = torch.tensor(self.V_static_train, dtype=torch.float32).to(device)
        vo_tr  = torch.tensor(self.v_o_train,      dtype=torch.float32).to(device)
        V_D    = torch.tensor(self.V_D,            dtype=torch.float32).to(device)
        tid_tr = torch.tensor(self.temporal_train,  dtype=torch.long).to(device)
        y_t    = torch.tensor(y_idx,               dtype=torch.long).to(device)

        # Optimizador
        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # Scheduler opcional: coseno desde lr hasta lr/100 a lo largo de todas las épocas
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_epochs, eta_min=self.lr / 100
            )
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        logger.info(
            "Entrenamiento — %d épocas | batch=%d | optimizer=%s | lr=%g | scheduler=%s | "
            "%d muestras | N=%d destinos",
            self.n_epochs, self.batch_size, self.optimizer_type,
            self.lr, "cosine" if scheduler else "none",
            n_train, len(self.station_index),
        )

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            rng     = np.random.default_rng(self.random_state + epoch)
            indices = rng.permutation(n_train)
            epoch_losses = []

            for start in range(0, n_train, self.batch_size):
                idx_np = indices[start : start + self.batch_size]
                idx    = torch.tensor(idx_np, dtype=torch.long)

                logits = self.model(Vs_tr[idx.to(device)], vo_tr[idx.to(device)],
                                    V_D, tid_tr[idx.to(device)])

                # Enmascarar la estación de subida (CPU → mascara sobre logits en device)
                oi       = self.origin_idx_train[idx_np]   # (batch,) numpy
                valid_oi = oi >= 0
                if valid_oi.any():
                    rows = np.where(valid_oi)[0]
                    logits[rows, oi[valid_oi]] = float('-inf')

                loss   = criterion(logits, y_t[idx.to(device)])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            if scheduler is not None:
                scheduler.step()

            mean_loss = float(np.mean(epoch_losses))
            self.train_losses.append(mean_loss)
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
                logger.info("Época %3d/%d — loss: %.4f | lr: %.2e", epoch, self.n_epochs, mean_loss, current_lr)
                print(f"      Época {epoch:3d}/{self.n_epochs} — loss: {mean_loss:.4f}  lr: {current_lr:.2e}")
            else:
                logger.info("Época %3d/%d — loss: %.4f", epoch, self.n_epochs, mean_loss)
                print(f"      Época {epoch:3d}/{self.n_epochs} — loss: {mean_loss:.4f}")

    # ------------------------------------------------------------------
    # Paso 9 — Evaluación en test
    # ------------------------------------------------------------------

    def _evaluate(self) -> None:
        """
        Evalúa el modelo sobre el conjunto de test.

        Métricas:
          - top-1 accuracy: el destino con mayor score coincide con el real
          - top-5 accuracy: el destino real aparece entre los 5 con mayor score
        """
        self.model.eval()

        station_to_idx = {s: i for i, s in enumerate(self.station_index)}
        y_idx = self.y_mdlf_test.map(station_to_idx).fillna(-1).astype(int).values

        device = getattr(self, '_device', torch.device('cpu'))
        Vs_te  = torch.tensor(self.V_static_test, dtype=torch.float32).to(device)
        vo_te  = torch.tensor(self.v_o_test,      dtype=torch.float32).to(device)
        V_D    = torch.tensor(self.V_D,           dtype=torch.float32).to(device)
        tid_te = torch.tensor(self.temporal_test,  dtype=torch.long).to(device)

        with torch.no_grad():
            logits = self.model(Vs_te, vo_te, V_D, tid_te)   # (n_test, N)

            # Enmascarar la estación de subida: nunca puede ser destino predicho
            oi       = self.origin_idx_test   # (n_test,) numpy
            valid_oi = oi >= 0
            if valid_oi.any():
                rows = np.where(valid_oi)[0]
                logits[rows, oi[valid_oi]] = float('-inf')

        top5_idx = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices.cpu().numpy()
        y_true   = y_idx  # numpy array

        top1 = float((top5_idx[:, 0] == y_true).mean())
        top5 = float(
            np.any(top5_idx == y_true[:, np.newaxis], axis=1).mean()
        )

        self.test_accuracy  = top1
        self.test_accuracy5 = top5
        # Guardar predicciones para análisis posterior (confusion matrix, etc.)
        self._y_true_eval = y_true                # índices de estación verdadera
        self._y_pred_eval = top5_idx[:, 0]        # índices top-1 predichos

        # Mover modelo a CPU para que attribution_by_group funcione sin cambios
        self.model = self.model.cpu()
        self._device = torch.device('cpu')
        logger.info(
            "Test accuracy — top-1: %.4f | top-5: %.4f | n_test=%d",
            top1, top5, len(y_true),
        )
        print(f"\n      ── Resultados MDLF ──────────────────────")
        print(f"      Top-1 accuracy : {top1:.4f}  ({top1:.1%})")
        print(f"      Top-5 accuracy : {top5:.4f}  ({top5:.1%})")
        print(f"      n_test         : {len(y_true)}")

    # ------------------------------------------------------------------
    # Interpretabilidad — atribución por grupo de features
    # ------------------------------------------------------------------

    def attribution_by_group(self) -> dict:
        """
        Calcula la importancia media de cada grupo de features mediante
        atribución por gradiente sobre el conjunto de test.

        Se calculan gradientes respecto a todas las entradas del modelo:

            grad_Vs[i, k, d] = ∂logits[i,j] / ∂V_static[i,k,d]   → Individual / Crowd / NBCrowd
            grad_VD[k, d]    = ∂logits[i,j] / ∂V_D[k,d]           → LDA / LINE  (pass separado)
            grad_et[i, d]    = ∂logits[i,j] / ∂e_t[i,d]           → Temporal    (forward hook en E_table)

        La importancia de cada entrada se estima como el valor absoluto del
        gradiente promediado sobre viajes y estaciones.

        Returns
        -------
        dict con claves:
            'por_dimension'          : np.ndarray (D_static,) — importancia por dim de V_static
            'individual'             : float
            'crowd'                  : float
            'neighborhood_crowd'     : float
            'lda'                    : float
            'line'                   : float
            'temporal'               : float
            'individual_dims'        : np.ndarray (D_ind,)
            'crowd_dims'             : np.ndarray (D_crowd,)
            'neighborhood_crowd_dims': np.ndarray (D_nbcrowd,)
            'lda_dims'               : np.ndarray (K,)
            'line_dims'              : np.ndarray (L,)
            'sub_feature_names'      : list[str]
        """
        if self.model is None or self.V_static_test is None:
            raise RuntimeError("Ejecuta run() antes de attribution_by_group().")

        self.model.eval()
        self.model.zero_grad()   # limpiar grads residuales del entrenamiento

        station_to_idx = {s: i for i, s in enumerate(self.station_index)}
        y_idx = self.y_mdlf_test.map(station_to_idx).fillna(-1).astype(int).values
        valid  = y_idx >= 0

        vo_np  = self.v_o_test[valid]
        tid_np = self.temporal_test[valid]
        y_t    = torch.tensor(y_idx[valid], dtype=torch.long)
        n_valid = int(valid.sum())

        # ── Pass 1: gradiente sobre V_static + temporal ───────────────────────
        # Temporal: usamos un forward hook sobre E_table para obtener
        # ∂loss/∂e_t  (gradiente respecto al OUTPUT del embedding, shape (n, D_T))
        # en lugar de ∂loss/∂E_table.weight (gradiente respecto a los parámetros).
        # Esto hace la atribución temporal comparable con la de V_static / V_D.
        has_od = self.model.od_module is not None
        _e_t_ref = [None]   # contenedor mutable para el closure del hook

        def _et_hook(module, inp, out):
            # out: (n, D_T) — salida del embedding, desconectada del grafo de pesos
            e_t = out.detach().clone().requires_grad_(True)
            _e_t_ref[0] = e_t
            return e_t   # inyectamos un leaf tensor con requires_grad

        _hook_handle = None
        if has_od:
            _hook_handle = self.model.od_module.E_table.register_forward_hook(_et_hook)

        Vs  = torch.tensor(self.V_static_test[valid], dtype=torch.float32, requires_grad=True)
        vo  = torch.tensor(vo_np,  dtype=torch.float32)
        V_D = torch.tensor(self.V_D, dtype=torch.float32)
        tid = torch.tensor(tid_np,   dtype=torch.long)

        logits = self.model(Vs, vo, V_D, tid)
        logits[torch.arange(n_valid), y_t].sum().backward()

        if _hook_handle is not None:
            _hook_handle.remove()

        # V_static importance
        grad      = Vs.grad.abs().detach().numpy()   # (n_valid, N, D_static)
        importance = grad.mean(axis=(0, 1))           # (D_static,)

        # Temporal importance: gradiente respecto a e_t (n_valid, D_T)
        # — misma espacialidad que V_static: promedio sobre viajes × dimensiones
        if has_od and _e_t_ref[0] is not None and _e_t_ref[0].grad is not None:
            imp_temporal = float(_e_t_ref[0].grad.abs().mean().item())
        else:
            imp_temporal = 0.0

        # ── Pass 2: gradiente sobre V_D (region features LDA + LINE) ──────────
        if has_od:
            self.model.zero_grad()
            V_D2  = torch.tensor(self.V_D, dtype=torch.float32, requires_grad=True)
            Vs2   = torch.tensor(self.V_static_test[valid], dtype=torch.float32)
            vo2   = torch.tensor(vo_np,  dtype=torch.float32)
            tid2  = torch.tensor(tid_np,  dtype=torch.long)

            logits2 = self.model(Vs2, vo2, V_D2, tid2)
            logits2[torch.arange(n_valid), y_t].sum().backward()

            grad_VD  = V_D2.grad.abs().detach().numpy()   # (N, D_R)
            imp_VD   = grad_VD.mean(axis=0) / n_valid     # (D_R,) — dividir por n_valid para
            # ser comparable con imp_individual: V_D es compartida entre todos los trips
            # y su gradiente acumula la suma de n_valid contribuciones, mientras que
            # V_static es per-trip y su gradiente ya se promedia sobre n_valid en mean(axis=(0,1))

            K = self.theta_c.shape[1] if self.theta_c is not None else 0
            L = (self.line_c.shape[1]
                 if (self.line_c is not None and self.line_c.shape[1] > 0) else 0)
            imp_lda  = imp_VD[:K]
            imp_line = imp_VD[K:K + L]
        else:
            K, L    = 0, 0
            imp_lda  = np.array([], dtype=np.float32)
            imp_line = np.array([], dtype=np.float32)

        # ── Separar bloques de V_static ────────────────────────────────────────
        D_ind     = getattr(self, "_D_ind_eff",     8 if self.include_prox else 6)
        D_crowd   = getattr(self, "_D_crowd_eff",   self.n_weeks)
        D_nbcrowd = getattr(self, "_D_nbcrowd_eff",
                            self.n_weeks if self.include_neighborhood_crowd else 0)

        imp_ind     = importance[:D_ind]
        imp_crowd   = importance[D_ind : D_ind + D_crowd]
        imp_nbcrowd = importance[D_ind + D_crowd : D_ind + D_crowd + D_nbcrowd]

        if self.include_prox:
            names = ["td_a", "td_ap", "sd", "tsd_b", "tsd_bp", "tsi", "prox_tsi", "prox_td"]
        else:
            names = ["td_a", "td_ap", "sd", "tsd_b", "tsd_bp", "tsi"]

        result = {
            "por_dimension"          : importance,
            "individual"             : float(imp_ind.mean())     if D_ind     > 0 else 0.0,
            "crowd"                  : float(imp_crowd.mean())   if D_crowd   > 0 else 0.0,
            "neighborhood_crowd"     : float(imp_nbcrowd.mean()) if D_nbcrowd > 0 else 0.0,
            "lda"                    : float(imp_lda.mean())     if K         > 0 else 0.0,
            "line"                   : float(imp_line.mean())    if L         > 0 else 0.0,
            "temporal"               : imp_temporal,
            "individual_dims"        : imp_ind,
            "crowd_dims"             : imp_crowd,
            "neighborhood_crowd_dims": imp_nbcrowd,
            "lda_dims"               : imp_lda,
            "line_dims"              : imp_line,
            "sub_feature_names"      : names,
        }

        # ── Imprimir resumen ────────────────────────────────────────────────────
        all_vals = [v for v in [
            result["individual"], result["crowd"], result["neighborhood_crowd"],
            result["lda"], result["line"], result["temporal"],
        ] if v > 0 and np.isfinite(v)]
        max_imp = (max(all_vals) if all_vals else 1.0) + 1e-9

        def bar(val): return "█" * int(val / max_imp * 20) if np.isfinite(val) else "NaN"

        print("\n      ── Atribución por grupo de features ────────")
        print(f"      n_viajes analizados : {n_valid}")
        if D_ind > 0:
            print(f"      Bloque INDIVIDUAL       : {result['individual']:.5f}  {bar(result['individual'])}")
            for name, val in zip(names, imp_ind):
                print(f"        {name:<12} {val:.5f}  {bar(val)}")
        if D_crowd > 0:
            print(f"      Bloque CROWD            : {result['crowd']:.5f}  {bar(result['crowd'])}")
            for w, val in enumerate(imp_crowd):
                print(f"        semana -{w+1:<5}  {val:.5f}  {bar(val)}")
        if D_nbcrowd > 0:
            print(f"      Bloque NEIGHBORHOOD      : {result['neighborhood_crowd']:.5f}  {bar(result['neighborhood_crowd'])}")
            for w, val in enumerate(imp_nbcrowd):
                print(f"        nb semana -{w+1:<4} {val:.5f}  {bar(val)}")
        if K > 0:
            print(f"      Bloque REGION (LDA)     : {result['lda']:.5f}  {bar(result['lda'])}")
        if L > 0:
            print(f"      Bloque REGION (LINE)    : {result['line']:.5f}  {bar(result['line'])}")
        if has_od:
            print(f"      Bloque TEMPORAL         : {result['temporal']:.5f}  {bar(result['temporal'])}")

        logger.info(
            "Atribución — individual: %.5f | crowd: %.5f | neighborhood: %.5f "
            "| lda: %.5f | line: %.5f | temporal: %.5f",
            result["individual"], result["crowd"], result["neighborhood_crowd"],
            result["lda"], result["line"], result["temporal"],
        )
        self.attribution = result
        return result

    # ------------------------------------------------------------------
    # Análisis de predicciones — confusion matrix + estadísticas
    # ------------------------------------------------------------------

    def prediction_analysis(self, top_n: int = 10, log: bool = False) -> dict:
        """
        Analiza las predicciones del modelo sobre el conjunto de test.

        Devuelve y opcionalmente loguea/imprime:
          · confusion_matrix  : pd.DataFrame (estaciones × estaciones) con conteos
          · per_station       : pd.DataFrame — por estación: n_real, n_hit, acc, top_error
          · most_predicted    : pd.Series   — estaciones más predichas (% sobre total pred)
          · most_failed       : pd.DataFrame — pares (real, predicho) más frecuentes en fallos

        Parameters
        ----------
        top_n : int
            Número de entradas a mostrar en los rankings.
        log : bool
            Si True, escribe el resumen también en el log del run.
        """
        if self._y_true_eval is None or self._y_pred_eval is None:
            raise RuntimeError("Ejecuta run() antes de prediction_analysis().")

        idx2name = {i: s for i, s in enumerate(self.station_index)}
        y_true_names = [idx2name[i] for i in self._y_true_eval]
        y_pred_names = [idx2name[i] for i in self._y_pred_eval]
        stations_in_test = sorted(set(y_true_names) | set(y_pred_names))

        # ── Confusion matrix ─────────────────────────────────────────────
        from sklearn.metrics import confusion_matrix as _cm
        labels = sorted(set(y_true_names))   # solo estaciones reales en test
        cm = _cm(y_true_names, y_pred_names, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        # ── Por estación: accuracy + principal error ──────────────────────
        rows = []
        for i, station in enumerate(labels):
            mask = np.array(y_true_names) == station
            if mask.sum() == 0:
                continue
            preds = np.array(y_pred_names)[mask]
            n_real = int(mask.sum())
            n_hit  = int((preds == station).sum())
            acc_s  = n_hit / n_real
            wrong  = preds[preds != station]
            top_err = pd.Series(wrong).value_counts()
            top_err_str = f"{top_err.index[0]} ({top_err.iloc[0]})" if len(top_err) > 0 else "—"
            rows.append({
                "estacion"  : station,
                "n_real"    : n_real,
                "n_acierto" : n_hit,
                "accuracy"  : round(acc_s, 4),
                "top_error" : top_err_str,
            })
        per_station = (
            pd.DataFrame(rows)
            .sort_values("n_real", ascending=False)
            .reset_index(drop=True)
        )

        # ── Estaciones más predichas ──────────────────────────────────────
        pred_counts = pd.Series(y_pred_names).value_counts()
        most_predicted = (pred_counts / len(y_pred_names) * 100).round(2)

        # ── Pares de error más frecuentes ────────────────────────────────
        err_mask = np.array(y_true_names) != np.array(y_pred_names)
        err_true = np.array(y_true_names)[err_mask]
        err_pred = np.array(y_pred_names)[err_mask]
        err_pairs = pd.Series(
            [f"{t}  →  {p}" for t, p in zip(err_true, err_pred)]
        ).value_counts()
        err_pct = (err_pairs / err_mask.sum() * 100).round(2)
        most_failed = pd.DataFrame({
            "par_real_predicho" : err_pairs.index,
            "n_errores"         : err_pairs.values,
            "pct_sobre_fallos"  : err_pct.values,
        }).head(top_n)

        # ── Imprimir resumen ──────────────────────────────────────────────
        lines = []
        lines.append("\n[Pred] Análisis de predicciones...")
        lines.append(f"\n      ── Estaciones más predichas (top {top_n}) ─────────────────")
        for est, pct in most_predicted.head(top_n).items():
            hit_pct = per_station.loc[per_station["estacion"] == est, "accuracy"]
            hit_str = f"  acc={float(hit_pct.iloc[0]):.1%}" if len(hit_pct) > 0 else ""
            lines.append(f"      {est:<40} {pct:>5.1f}%{hit_str}")

        lines.append(f"\n      ── Estaciones con menor accuracy (mín. 5 viajes) ──────────")
        low_acc = (
            per_station[per_station["n_real"] >= 5]
            .sort_values("accuracy")
            .head(top_n)
        )
        for _, r in low_acc.iterrows():
            lines.append(
                f"      {r['estacion']:<40} acc={r['accuracy']:.1%}  "
                f"({r['n_acierto']}/{r['n_real']})  error→ {r['top_error']}"
            )

        lines.append(f"\n      ── Pares de error más frecuentes (top {top_n}) ───────────")
        for _, r in most_failed.iterrows():
            lines.append(
                f"      {r['par_real_predicho']:<60} "
                f"n={r['n_errores']}  ({r['pct_sobre_fallos']:.1f}% de fallos)"
            )

        output = "\n".join(lines)
        print(output)

        if log:
            for line in lines:
                logger.info(line.strip())

        return {
            "confusion_matrix" : cm_df,
            "per_station"      : per_station,
            "most_predicted"   : most_predicted,
            "most_failed"      : most_failed,
        }

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_train = len(self.train_c) if self.train_c is not None else "?"
        n_test  = len(self.test_c)  if self.test_c  is not None else "?"
        acc     = f"{self.test_accuracy:.4f}" if self.test_accuracy is not None else "pending"
        acc5 = f"{self.test_accuracy5:.4f}" if self.test_accuracy5 is not None else "pending"
        return (
            f"PipelineMDLF("
            f"train={n_train}, test={n_test}, "
            f"n_weeks={self.n_weeks}, n_heads={self.n_heads}, "
            f"C={self.cnn_channels}, M={self.kernel_size}, "
            f"optimizer={self.optimizer_type}, lr={self.lr}, "
            f"n_epochs={self.n_epochs}, "
            f"top1={acc}, top5={acc5})"
        )
