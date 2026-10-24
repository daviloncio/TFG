#!/usr/bin/env python3
"""
run_lda.py — Pipeline standalone para RegionFunctionLDA.

Carga todos los ficheros matriz_od.csv (y/o .parquet) de un directorio,
ajusta el LDA y guarda Theta + inspección de tópicos en output/.

Uso básico
----------
    python run_lda.py \\
        --matrices_dir ../../cubo_izquierdo/matrices_OD \\
        --n_topics 16 \\
        --slot_minutes 15

Con festivos
------------
    python run_lda.py \\
        --matrices_dir ../../cubo_izquierdo/matrices_OD \\
        --n_topics 16 \\
        --holiday_dates 2025-01-01 2025-01-06 2025-04-18

Ajuste de alpha/beta
--------------------
    python run_lda.py \\
        --matrices_dir ../../cubo_izquierdo/matrices_OD \\
        --n_topics 32 \\
        --slot_minutes 30 \\
        --alpha 0.1 \\
        --beta 0.01
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Para poder ejecutar desde cualquier directorio
sys.path.insert(0, str(Path(__file__).parent))
from region_function_lda import RegionFunctionLDA  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_od_matrices(matrices_dir: str | Path) -> pd.DataFrame:
    """
    Carga y concatena todos los ficheros matriz_od.csv / matriz_od.parquet
    encontrados recursivamente bajo matrices_dir.

    Nota sobre duplicados
    ---------------------
    Si existe tanto .csv como .parquet para la misma carpeta, se cargarán
    ambos y los conteos se duplicarán. Evita esto apuntando a una carpeta
    que solo contenga un formato por día, o usa --only_csv / --only_parquet.

    Para LDA, duplicar counts no cambia las distribuciones de tópicos
    (LDA trabaja con proporciones relativas dentro de cada documento),
    pero sí infla el tiempo de cómputo innecesariamente.
    """
    root = Path(matrices_dir)
    csv_paths     = sorted(root.rglob("matriz_od.csv"))
    parquet_paths = sorted(root.rglob("matriz_od.parquet"))

    if not csv_paths and not parquet_paths:
        raise FileNotFoundError(
            f"No se encontró ningún fichero matriz_od.csv / .parquet bajo: {root}"
        )

    dfs = []
    for p in csv_paths:
        logger.info("Cargando CSV  : %s", p.relative_to(root))
        dfs.append(pd.read_csv(p))

    for p in parquet_paths:
        logger.info("Cargando parquet: %s", p.relative_to(root))
        dfs.append(pd.read_parquet(p))

    df = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Total cargado: %d filas de %d ficheros",
        len(df), len(csv_paths) + len(parquet_paths),
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RegionFunctionLDA — extrae features de función regional por estación.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--matrices_dir",
        required=True,
        help="Ruta a la carpeta matrices_OD (se buscan subdirectorios recursivamente).",
    )
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "output"),
        help="Carpeta donde guardar theta.parquet y topics_inspection.csv.",
    )
    p.add_argument("--n_topics_metro",     type=int,   default=16,   help="K para Metro (02).")
    p.add_argument("--n_topics_cercanias", type=int,   default=8,    help="K para Cercanías (04).")
    p.add_argument("--slot_minutes",       type=int,   default=15,   help="Tamaño del bin temporal (15 ó 30).")
    p.add_argument("--alpha",              type=float, default=None, help="Prior doc-topic (None = 1/K).")
    p.add_argument("--beta",               type=float, default=None, help="Prior topic-word (None = 1/K).")
    p.add_argument("--max_iter",           type=int,   default=20,   help="Iteraciones del EM variacional.")
    p.add_argument(
        "--n_top_tokens",
        type=int,
        default=15,
        help="Tokens a mostrar por tópico en la inspección.",
    )
    p.add_argument(
        "--holiday_dates",
        nargs="*",
        default=[],
        metavar="YYYY-MM-DD",
        help="Fechas a tratar como fin de semana (festivos).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_red(
    od_df: pd.DataFrame,
    prefijo: str,
    nombre: str,
    n_topics: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple:
    """Entrena un modelo LDA para una red (metro o cercanías) y guarda resultados."""
    model = RegionFunctionLDA(
        n_topics=n_topics,
        slot_minutes=args.slot_minutes,
        alpha=args.alpha,
        beta=args.beta,
        holiday_dates=args.holiday_dates if args.holiday_dates else None,
        filter_prefixes=[prefijo],
        max_iter=args.max_iter,
    )
    logger.info("--- %s (prefijo %s) | K=%d ---", nombre, prefijo, n_topics)
    theta = model.fit_transform(od_df)

    # Guardar Theta
    theta.to_parquet(output_dir / f"theta_{prefijo}.parquet")
    theta.round(5).to_csv(output_dir / f"theta_{prefijo}.csv")

    # Guardar inspección de tópicos
    model.inspect_topics(args.n_top_tokens).to_csv(
        output_dir / f"topics_inspection_{prefijo}.csv"
    )
    logger.info("Theta %s guardado → shape %s", nombre, theta.shape)
    return model, theta


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos históricos (una sola vez para ambas redes)
    od_df = load_od_matrices(args.matrices_dir)

    # 2. Modelo Metro (02)
    model_metro, theta_metro = _run_red(
        od_df, prefijo='02', nombre='Metro',
        n_topics=args.n_topics_metro, args=args, output_dir=output_dir,
    )

    # 3. Modelo Cercanías (04)
    model_cercanias, theta_cercanias = _run_red(
        od_df, prefijo='04', nombre='Cercanías',
        n_topics=args.n_topics_cercanias, args=args, output_dir=output_dir,
    )

    # 4. Resumen por consola
    for nombre, model, theta in [
        ('Metro (02)', model_metro, theta_metro),
        ('Cercanías (04)', model_cercanias, theta_cercanias),
    ]:
        _print_summary(nombre, model, theta, args)


def _print_summary(
    nombre: str,
    model: RegionFunctionLDA,
    theta: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"RED: {nombre}  —  TÓPICOS LATENTES")
    print(sep)
    print(model.topic_summary(n_top_tokens=8))

    print(f"\n{sep}")
    print(f"RED: {nombre}  —  THETA primeras 15 estaciones")
    print(sep)
    print(theta.round(3).head(15).to_string())

    print(f"\n{sep}")
    print(f"RED: {nombre}  —  ESTACIÓN MÁS REPRESENTATIVA POR TÓPICO")
    print(sep)
    for col in theta.columns:
        best_station = theta[col].idxmax()
        best_val = theta[col].max()
        print(f"  {col}: {best_station}  ({best_val:.3f})")

    print(f"\n{sep}")
    print(f"Parámetros: slot={args.slot_minutes}min | alpha={args.alpha} | "
          f"beta={args.beta} | max_iter={args.max_iter}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print(sep)


if __name__ == "__main__":
    main()
