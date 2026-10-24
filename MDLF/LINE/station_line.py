"""
station_line.py
===============
Extrae embeddings de posición para cada estación usando LINE
(Large-scale Information Network Embedding), tal como describe
el paper MDLF (Zhao et al., 2022) para las "region remoteness features".

Idea central
------------
La posición de una estación en la red se captura mediante su
accesibilidad temporal respecto al resto de estaciones.
LINE aprende embeddings que preservan dos tipos de proximidad:

  - First-order:  estaciones directamente conectadas → embeddings cercanos.
                  Si no hay arista directa, first-order proximity = 0.
  - Second-order: estaciones con vecindarios similares → embeddings cercanos,
                  aunque no estén directamente conectadas entre sí.

El embedding final de cada estación es la concatenación de ambos:
    e_i = [e_i^(fo) || e_i^(so)]   →  2 * dim dimensiones

Pesos del grafo
---------------
Los pesos de las aristas son tiempos de viaje (minutos).
Internamente se convierten a similitud: sim = 1 / tiempo,
de modo que estaciones más cercanas tienen mayor peso en LINE.

Uso
---
    model = StationLINE(dim=32, num_epochs=50)
    emb_station = model.fit_transform(G)      # G: nx.MultiDiGraph
    emb_ap      = model.to_punto_acceso(G)    # expandido a punto_acceso

    # emb_station : pd.DataFrame [n_estaciones  × 64]
    # emb_ap      : pd.DataFrame [n_acc_points  × 64]
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StationLINE:
    """
    LINE embedding (first-order + second-order) para estaciones de transporte.

    Parameters
    ----------
    dim : int
        Dimensión de cada embedding parcial (first-order y second-order).
        El embedding final tiene 2*dim columnas.
    num_negative : int
        Muestras negativas por par positivo en el SGD.
    lr : float
        Tasa de aprendizaje inicial del SGD (decae linealmente a lr*1e-4).
    num_epochs_fo : int
        Épocas de entrenamiento para first-order proximity.
        FO converge rápido en grafos sparse — valor recomendado: 200.
    num_epochs_so : int
        Épocas de entrenamiento para second-order proximity.
        SO necesita más pasos para aprender vecindarios — valor recomendado: 1000.
    init_scale : float
        Desviación estándar de la inicialización gaussiana de los embeddings.
        Un valor mayor (ej. 0.1) genera pasos de gradiente más grandes desde
        el inicio (el update es g * u_j, proporcional a |u_j|), lo que ayuda
        a que FO converja en grafos muy sparse. Con 0.01 los embeddings apenas
        se mueven del origen con pocas épocas.
    asymmetric : bool
        Si True, conserva la naturaleza dirigida del grafo original: NO se
        simetriza la matriz de adyacencia y first-order proximity entrena sobre
        todos los edges dirigidos (i→j y j→i por separado).
        Esto triplica la señal disponible respecto al modo simétrico
        (370 aristas dirigidas vs 96 pares no dirigidos en cercanías),
        pero rompe la suposición de simetría del paper LINE original.
        Si False (por defecto), se promedia t_ij y t_ji y se entrena sobre
        pares no dirigidos, siguiendo el paper.
    random_state : int
    """

    def __init__(
        self,
        dim: int = 32,
        num_negative: int = 1,
        lr: float = 0.025,
        num_epochs_fo: int = 200,
        num_epochs_so: int = 1000,
        init_scale: float = 0.1,
        asymmetric: bool = False,
        random_state: int = 42,
    ) -> None:
        self.dim = dim
        self.num_negative = num_negative
        self.lr = lr
        self.num_epochs_fo = num_epochs_fo
        self.num_epochs_so = num_epochs_so
        self.init_scale = init_scale
        self.asymmetric = asymmetric
        self.random_state = random_state

        # Estado interno — solo existe tras fit_transform
        self.station_index_: list[str] | None = None
        self.embeddings_: pd.DataFrame | None = None
        self.adj_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit_transform(self, G: nx.MultiDiGraph) -> pd.DataFrame:
        """
        Entrena LINE sobre el grafo G y devuelve los embeddings concatenados.

        Parameters
        ----------
        G : nx.MultiDiGraph
            Grafo de la red de transporte. Nodos = estaciones físicas.
            Aristas con atributo 'weight' = tiempo de viaje en minutos.

        Returns
        -------
        pd.DataFrame  [n_estaciones × 2*dim]
            Columnas: fo_0 … fo_{dim-1}  (first-order)
                      so_0 … so_{dim-1}  (second-order)
            Índice:   nombre de estación (nodo del grafo)
        """
        rng = np.random.default_rng(self.random_state)

        nodes, adj = self._build_adjacency(G)
        self.station_index_ = nodes
        self.adj_ = adj
        n = len(nodes)

        if self.asymmetric:
            n_edges_fo = int((adj > 0).sum())
        else:
            n_edges_fo = int((np.triu(adj, k=1) > 0).sum())
        n_edges_so = int((adj > 0).sum())
        logger.info(
            "LINE: %d nodos | %d aristas first-order | %d aristas second-order",
            n, n_edges_fo, n_edges_so,
        )

        mode_tag = "dirigido" if self.asymmetric else "no dirigido"
        print(f"Entrenando first-order  ({n_edges_fo} aristas {mode_tag}, {self.num_epochs_fo} épocas)...")
        emb_fo = self._train_first_order(adj, n, rng)

        print(f"Entrenando second-order ({n_edges_so} aristas dirigidas, {self.num_epochs_so} épocas)...")
        emb_so = self._train_second_order(adj, n, rng)

        emb = np.concatenate([emb_fo, emb_so], axis=1)
        cols = [f"fo_{i}" for i in range(self.dim)] + [f"so_{i}" for i in range(self.dim)]
        self.embeddings_ = pd.DataFrame(emb, index=nodes, columns=cols)

        print(f"Completado: {n} estaciones × {2 * self.dim} dimensiones")
        return self.embeddings_

    def to_punto_acceso(self, G: nx.MultiDiGraph) -> pd.DataFrame:
        """
        Expande embeddings de nivel estación a nivel punto_acceso.

        Cada estación puede tener uno o varios access_point_codes
        (ej. una estación en varias líneas). Todos los access points
        de una misma estación reciben el mismo vector LINE.

        Parameters
        ----------
        G : nx.MultiDiGraph
            El mismo grafo usado en fit_transform().

        Returns
        -------
        pd.DataFrame  [n_puntos_acceso × 2*dim]
            Índice: punto_acceso (ej. '02_L1_P53')
        """
        self._check_fitted()
        rows: list[tuple[str, np.ndarray]] = []
        for node, data in G.nodes(data=True):
            if node not in self.embeddings_.index:
                continue
            aps = data.get("access_point_codes", set())
            if isinstance(aps, str):
                aps = {aps}
            vec = self.embeddings_.loc[node].values
            for ap in aps:
                rows.append((ap, vec))

        if not rows:
            return pd.DataFrame(columns=self.embeddings_.columns)

        index = [r[0] for r in rows]
        values = np.stack([r[1] for r in rows])
        return pd.DataFrame(values, index=index, columns=self.embeddings_.columns)

    # ------------------------------------------------------------------
    # Construcción de la matriz de adyacencia
    # ------------------------------------------------------------------

    def _build_adjacency(self, G: nx.MultiDiGraph) -> tuple[list[str], np.ndarray]:
        """
        Construye una matriz de adyacencia sparse usando los edges originales
        del grafo topológico (conexiones directas entre estaciones adyacentes).

        Para cada par de nodos conectados por al menos una arista:
          1. Tomar el tiempo mínimo entre todas las aristas i→j (puede haber
             varias líneas compartiendo tramo).
          2. Simetrizar: tiempo_avg = (t_ij + t_ji) / 2 cuando existen ambas
             direcciones; si solo existe una, se usa esa.
          3. Convertir a similitud: sim = 1 / tiempo
          4. Diagonal = 0 (sin auto-bucles)

        Resultado: grafo sparse con ~nº_edges_originales pares conectados.
        LINE tiene señal clara sobre qué estaciones son adyacentes, sin
        diluirse por conexiones lejanas de baja similitud.

        Returns
        -------
        nodes : list[str]  — nodos ordenados
        adj   : np.ndarray [n × n]  — simétrica, valores ≥ 0
        """
        nodes = sorted(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        # Tiempo mínimo directo entre pares adyacentes (varias líneas → min)
        T = np.zeros((n, n))
        for u, v, data in G.edges(data=True):
            i, j = idx[u], idx[v]
            w = data.get("weight", 1.0)
            if w > 0:
                if T[i, j] == 0:
                    T[i, j] = w
                else:
                    T[i, j] = min(T[i, j], w)

        # Simetrizar: promedio cuando existen ambas direcciones
        both    = (T > 0) & (T.T > 0)
        only_ij = (T > 0) & (T.T == 0)
        only_ji = (T == 0) & (T.T > 0)

        T_avg = np.zeros((n, n))
        T_avg[both]    = (T[both] + T.T[both]) / 2.0
        T_avg[only_ij] = T[only_ij]
        T_avg[only_ji] = T.T[only_ji]

        if self.asymmetric:
            # Modo dirigido: conservar pesos asimétricos originales (sin promediar)
            with np.errstate(divide="ignore", invalid="ignore"):
                sim_dir = np.where(T > 0, 1.0 / T, 0.0)
            np.fill_diagonal(sim_dir, 0.0)

            n_edges_dir = int((sim_dir > 0).sum())
            n_possible_dir = n * (n - 1)
            logger.info(
                "Directed adjacency: %d/%d aristas dirigidas (%.1f%%)",
                n_edges_dir, n_possible_dir, n_edges_dir / n_possible_dir * 100,
            )
            return nodes, sim_dir

        # Modo simétrico (por defecto): convertir a similitud y forzar simetría exacta
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(T_avg > 0, 1.0 / T_avg, 0.0)
        T_sym = (sim + sim.T) / 2.0
        np.fill_diagonal(T_sym, 0.0)

        n_connected = int((np.triu(T_sym, k=1) > 0).sum())
        n_possible = n * (n - 1) // 2
        logger.info(
            "Sparse adjacency: %d/%d pares conectados (%.1f%%)",
            n_connected, n_possible, n_connected / n_possible * 100,
        )

        return nodes, T_sym

    # ------------------------------------------------------------------
    # Entrenamiento LINE
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _noise_distribution(self, adj: np.ndarray) -> np.ndarray:
        """Distribución de ruido proporcional a grado^0.75 (como word2vec)."""
        degree = adj.sum(axis=1)
        noise = np.power(np.maximum(degree, 1e-10), 0.75)
        return noise / noise.sum()

    def _train_first_order(
        self, adj: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        First-order proximity: pares directamente conectados → embeddings cercanos.

        Un único espacio de embeddings U [n × dim].
        Por cada arista no dirigida (i,j) con peso w:
          Loss = -w * log σ(u_i · u_j)
                 - w * Σ_k log σ(-u_i · u_k)   [k: muestras negativas]

        Actualización SGD con lr decreciente (paper LINE, Tang et al. 2015):
          lr_t = max(lr_0 * (1 - t / T), lr_0 * 1e-4)
          donde t = paso actual, T = total de pasos (épocas × aristas)

          u_i += lr_t * w * (1 - σ(u_i·u_j)) * u_j   [positivo]
          u_j += lr_t * w * (1 - σ(u_i·u_j)) * u_i
          u_i -= lr_t * w * σ(u_i·u_k) * u_k          [negativo]
          u_k -= lr_t * w * σ(u_i·u_k) * u_i
        """
        emb = rng.normal(0.0, self.init_scale, (n, self.dim))
        noise = self._noise_distribution(adj)

        # Aristas: dirigidas si asymmetric=True, no dirigidas (i < j) si False
        if self.asymmetric:
            ii, jj = np.where(adj > 0)
        else:
            ii, jj = np.where(np.triu(adj, k=1) > 0)
        weights = adj[ii, jj]
        order = np.arange(len(ii))

        total_steps = len(order) * self.num_epochs_fo
        lr_min = self.lr * 1e-4
        step = 0

        for epoch in range(self.num_epochs_fo):
            rng.shuffle(order)
            for idx in order:
                lr_t = max(self.lr * (1.0 - step / total_steps), lr_min)
                step += 1

                i, j, w = int(ii[idx]), int(jj[idx]), float(weights[idx])
                u_i = emb[i].copy()
                u_j = emb[j].copy()

                # Par positivo
                sig = self._sigmoid(np.dot(u_i, u_j))
                g = lr_t * w * (1.0 - sig)
                emb[i] += g * u_j
                emb[j] += g * u_i

                # Muestras negativas (usando snapshot u_i)
                for k in rng.choice(n, size=self.num_negative, p=noise):
                    u_k = emb[k].copy()
                    sig_neg = self._sigmoid(np.dot(u_i, u_k))
                    g_neg = lr_t * w * sig_neg
                    emb[i] -= g_neg * u_k
                    emb[k] -= g_neg * u_i

        return emb

    def _train_second_order(
        self, adj: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Second-order proximity: vecindarios similares → embeddings cercanos.

        Dos espacios de embeddings:
          U     [n × dim]  — embedding como nodo objetivo (target)
          U_ctx [n × dim]  — embedding como contexto

        Por cada arista dirigida (i→j) con peso w:
          Loss = -w * log σ(U_ctx[j] · U[i])
                 - w * Σ_k log σ(-U_ctx[k] · U[i])

        Devuelve solo U (target), que es el embedding final de second-order.
        U_ctx se descarta — solo sirve durante el entrenamiento.
        Aplica el mismo lr decay lineal que first-order (paper LINE).
        """
        U = rng.normal(0.0, self.init_scale, (n, self.dim))
        U_ctx = rng.normal(0.0, self.init_scale, (n, self.dim))
        noise = self._noise_distribution(adj)

        # Aristas dirigidas (ambos sentidos)
        ii, jj = np.where(adj > 0)
        weights = adj[ii, jj]
        order = np.arange(len(ii))

        total_steps = len(order) * self.num_epochs_so
        lr_min = self.lr * 1e-4
        step = 0

        for epoch in range(self.num_epochs_so):
            rng.shuffle(order)
            for idx in order:
                lr_t = max(self.lr * (1.0 - step / total_steps), lr_min)
                step += 1

                i, j, w = int(ii[idx]), int(jj[idx]), float(weights[idx])
                u_i = U[i].copy()
                u_j_ctx = U_ctx[j].copy()

                # Par positivo
                sig = self._sigmoid(np.dot(u_j_ctx, u_i))
                g = lr_t * w * (1.0 - sig)
                U[i] += g * u_j_ctx
                U_ctx[j] += g * u_i

                # Muestras negativas
                for k in rng.choice(n, size=self.num_negative, p=noise):
                    u_k_ctx = U_ctx[k].copy()
                    sig_neg = self._sigmoid(np.dot(u_k_ctx, u_i))
                    g_neg = lr_t * w * sig_neg
                    U[i] -= g_neg * u_k_ctx
                    U_ctx[k] -= g_neg * u_i

        return U

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.embeddings_ is None:
            raise RuntimeError(
                "Modelo no ajustado. Llama a fit_transform() primero."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.embeddings_ is not None else "not fitted"
        return (
            f"StationLINE(dim={self.dim}, num_negative={self.num_negative}, "
            f"lr={self.lr}, num_epochs_fo={self.num_epochs_fo}, "
            f"num_epochs_so={self.num_epochs_so}, init_scale={self.init_scale}, "
            f"asymmetric={self.asymmetric}, status={status})"
        )
