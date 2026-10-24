import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from check_transbordo import CheckTransbordo

buffer = pd.read_csv(BASE_DIR / "tiempos_viaje_pares.csv")
pares_existentes = set(buffer["id_par"].tolist())

aps_04L0 = set()
for id_par in pares_existentes:
    partes = id_par.split("_y_")
    if len(partes) == 2:
        o, d = partes
        if o.startswith("04_L0"):
            aps_04L0.add(o)
        if d.startswith("04_L0"):
            aps_04L0.add(d)

print(f"APs 04_L0 encontrados en buffer: {len(aps_04L0)}")

pares_faltantes = [
    (o, d)
    for o, d in product(aps_04L0, aps_04L0)
    if o != d and f"{o}_y_{d}" not in pares_existentes
]
print(f"Pares faltantes a calcular: {len(pares_faltantes)}")

ct = CheckTransbordo()

nuevas_filas = []
for i, (origen, destino) in enumerate(pares_faltantes):
    id_par = f"{origen}_y_{destino}"
    try:
        tiempo = ct.calculo_tiempo_estimado(origen, destino)
    except Exception:
        tiempo = np.nan
    nuevas_filas.append({"id_par": id_par, "tipo": tiempo})
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(pares_faltantes)} calculados...")

nans = sum(1 for r in nuevas_filas if pd.isna(r['tipo']))
print(f"Calculados: {len(nuevas_filas)}, con NaN: {nans}")

df_nuevos = pd.DataFrame(nuevas_filas)
df_final = pd.concat([buffer, df_nuevos], ignore_index=True)
df_final.to_csv(BASE_DIR / "tiempos_viaje_pares.csv", index=False)
print(f"tiempos_viaje_pares.csv actualizado con {len(df_final)} filas totales.")
