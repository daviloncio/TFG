# MDLF — Documentación del flujo end-to-end

Basado en: *Zhao et al., 2022 — "A Multi-View-Based Deep Learning Framework for Individual Trip Destination Prediction in Public Transportation Systems"*

---

## 1. Visión general

El módulo MDLF predice el **destino de viajes UNLINKED de cercanías** (viajes sin predicción de bajada en el pipeline principal). Es un problema de clasificación multiclase: dado un viaje de subida en la estación `s_o`, el modelo elige la estación de destino más probable entre las N estaciones de cercanías.

El modelo combina cuatro vistas complementarias del comportamiento del viajero:

| Vista | Módulo | Qué captura |
|---|---|---|
| Region Function | LDA + LINE | Función y conectividad de cada estación |
| Individual | `IndividualFeatureExtractor` | Historial de destinos del propio usuario |
| Crowd | `CrowdFeatureExtractor` | Historial colectivo de todos los usuarios desde el mismo origen y franja |
| Neighborhood Crowd *(opcional)* | `ResidenceInferrer` + `NeighborhoodCrowdExtractor` | Historial colectivo filtrado a usuarios del mismo barrio de residencia |
| Temporal | Embedding `E` entrenado | Contexto temporal del viaje (hora + tipo de día) |

El módulo se puede activar de dos formas:
- Desde `cubo_izquierdo/pipeline.py` con `20 in hipotesis` (requiere `nombre_almacen` en `mdlf_params`)
- Directamente desde un notebook instanciando `PipelineMDLF` con `nombre_almacen`

---

## 2. Entradas al módulo

`PipelineMDLF` **no recibe DataFrames en memoria**. Carga los datos desde el almacén generado previamente por la hipótesis 21 del pipeline de cubo izquierdo.

### Parámetro obligatorio

| Parámetro | Tipo | Descripción |
|---|---|---|
| `nombre_almacen` | `str` | Nombre de la subcarpeta en `MDLF/almacen/` que contiene los 5 parquets de entrada. Se genera con hipótesis 21, usando `Path(self.path).stem` como nombre. Ejemplo: `"50mil_tarjetas_3_al_5_marzo"` |

### Ficheros que carga desde `MDLF/almacen/<nombre_almacen>/`

| Fichero | Contenido original |
|---|---|
| `x_pred_checked.parquet` | Salida completa de `check_transbordo`. Historial para features individuales y crowd. Columnas clave: `id_tarjeta`, `punto_acceso`, `timestamp_validacion`, `tipo_transporte`, `prediccion_descenso` |
| `unknown.parquet` | Subidas con `prediccion_descenso == 'unknown'`. Candidatos a predecir con MDLF |
| `x_test.parquet` | Subidas con bajada real conocida, de `crear_mapeos` |
| `y_test.parquet` | Bajadas reales correspondientes a `x_test`, alineadas **por posición** |
| `subidas_sin_pred.parquet` | Solo Munizaga: viajes únicos en su día sin predicción (vacío si no aplica) |

### Cómo generar el almacén

```python
# Ejecutar una sola vez con hipótesis 21:
Pipeline(
    '50mil_tarjetas_3_al_5_marzo.csv',
    hipotesis={5, 10, 21},
    hiperparams=hip,
).start()
# → Crea MDLF/almacen/50mil_tarjetas_3_al_5_marzo/ con los 5 parquets
```

### Hiperparámetros (`mdlf_params`)

| Parámetro | Default | Descripción |
|---|---|---|
| `od_matrices_subdirs` | `None` | Lista de subcarpetas de matrices OD para LDA (relativas a `cubo_izquierdo/matrices/`) |
| `ttp_paths` | `None` | Lista de rutas a los 6 TTP CSVs para construir el historial individual con `IndividualHistBuilder`. Si es `None` y no hay caché, se usa `x_pred_checked` como fallback |
| `cubo_hiperparams` | `{}` | Hiperparámetros del pipeline cubo izquierdo (`dia_virtual`, `radio_busqueda`, etc.) necesarios para `IndividualHistBuilder` cuando construye el historial desde TTP CSVs |
| `n_topics_cercanias` | 16 | K: número de tópicos LDA para cercanías |
| `n_weeks` | 2 | Semanas de historial colectivo para crowd features |
| `n_heads` | 4 | H: cabezas de atención bilineal en OD Region Correlation |
| `d_t` | 16 | D_T: dimensión del embedding temporal E |
| `cnn_channels` | 16 | C: canales de las capas Conv1d |
| `kernel_size` | 1 | M: kernel sobre N estaciones (1 = independiente por estación) |
| `optimizer` | `'sgd'` | `'sgd'` (paper, momentum=0.9) o `'adamw'` |
| `lr` | 0.01 | Learning rate inicial |
| `batch_size` | 64 | Tamaño del batch en entrenamiento |
| `n_epochs` | 20 | Número de épocas |
| `lr_scheduler` | `False` | Si `True`, activa `CosineAnnealingLR`: el LR decae desde `lr` hasta `lr/100` a lo largo de las `n_epochs` épocas. Recomendado activar junto con `n_epochs` ≥ 50 |
| `include_prox` | `False` | Activar 5ª feature individual PROX (MDLF gen. 2) |
| `include_neighborhood_crowd` | `False` | Activar Neighborhood Crowd features |
| `dominance_ratio` | 2.0 | Ratio mínimo top/second para confirmar parada de residencia |
| `min_days_residence` | 5 | Días mínimos de historial para inferir residencia |
| `min_neighbors` | 10 | Usuarios mínimos en el vecindario para usar su señal |
| `test_size` | 0.2 | Fracción del conjunto de test |
| `random_state` | 42 | Semilla aleatoria |
| `ablation` | `None` | Conjunto de módulos a desactivar: `{'lda', 'line', 'individual', 'crowd', 'neighborhood_crowd'}` |

---

## 3. Construcción del conjunto de etiquetas (`_build_labeled_set`)

El modelo necesita viajes con **etiqueta real de destino** para entrenamiento supervisado.

```
unknown (cercanías)  ──┐
                        ├── concat + dedup ──→ unlinked_c
subidas_sin_pred    ──┘   (solo si Munizaga)

unlinked_c  ──INNER JOIN (id_tarjeta, timestamp_validacion)──→  labeled_c
                         x_test + y_test[punto_acceso]
```

**Detalles clave:**
- `x_test` e `y_test` están **alineados por posición**, no por clave de bajada. Se asigna `destino_real = y_test["punto_acceso"].values` antes del merge para evitar usar timestamps de bajada.
- Los timestamps de Spark (`unknown`) son tz-aware UTC; los de `x_test` son tz-naive. Se normaliza con `.dt.tz_localize(None)` en ambos lados antes del merge.
- Si `subidas_sin_pred` está activo (Munizaga), se concatena y se elimina duplicados por `(id_tarjeta, timestamp_validacion)`.

El resultado es `labeled_c`: DataFrame con los viajes de cercanías que tienen etiqueta real y pueden usarse para entrenar y evaluar el modelo.

---

## 4. Feature extraction — cuatro módulos

### 4.1 LDA — Region Function (`RegionFunctionLDA`)

**Archivo:** `MDLF/LDA/region_function_lda.py`

Aprende una representación de la **función de cada estación** a partir de las matrices OD (origen-destino). Usando Latent Dirichlet Allocation sobre los flujos históricos, asigna a cada estación un vector de K tópicos que codifica su rol funcional (residencial, laboral, comercial, etc.).

| Concepto | Valor |
|---|---|
| Datos de entrada | Matrices OD en formato long (parquet), de `cubo_izquierdo/matrices/<subdirs>/` |
| Período usado | Feb 2025: 20 días laborables + 8 fin de semana (28 días) |
| Filtro | Solo estaciones con prefijo `'04'` (cercanías) |
| **Salida: `theta_c`** | DataFrame `[N × K]`, índice = `punto_acceso` |
| **Dimensión por estación** | **K = 16** (por defecto) |

### 4.2 LINE — Station Embeddings (`StationLINE`)

**Archivo:** `MDLF/LINE/station_line.py`

Aprende embeddings de estaciones a partir del **grafo de red de cercanías**, preservando la similitud estructural entre estaciones conectadas. Combina first-order (FO) y second-order (SO) proximity.

| Concepto | Valor |
|---|---|
| Datos de entrada | Grafo de transbordo pkl de `check_transbordo` |
| Método | LINE FO+SO concatenados, aggregated por `punto_acceso` |
| **Salida: `line_c`** | DataFrame `[N × 64]`, índice = `punto_acceso` |
| **Dimensión por estación** | **L = 64** (32 FO + 32 SO) |

### 4.3 Features individuales (`IndividualHistBuilder` + `IndividualFeatureExtractor`)

#### Construcción del historial — `IndividualHistBuilder`

**Archivo:** `MDLF/INDIVIDUAL/hist_builder.py`

`IndividualFeatureExtractor` necesita viajes ya procesados (con `prediccion_descenso` y `tipo_transporte`) para saber adónde viajó cada usuario en el pasado. `IndividualHistBuilder` construye y cachea ese historial desde los TTP CSVs de los 6 períodos de 2025 (~81 GB).

**Prioridad de carga:**
1. Si existe `almacen/<nombre>/x_pred_checked_hist_indiv_mdlf.parquet` → carga directa
2. Si `ttp_paths` está definido → construye con `build()` (primera ejecución)
3. Fallback: usa `x_pred_checked` del período actual (aviso por consola)

**Flujo de `build()` — estrategia híbrida:**

Los datos del historial se construyen en dos grupos según si el destino se puede conocer exactamente o hay que predecirlo:

**Grupo A — Subidas con bajada real validada** (metro, cercanías, tranvía):  
La siguiente validación del usuario en el mismo día operativo es un código de descenso, del mismo modo de transporte, y en tiempo verosímil (≤2.5h cercanías, ≤1.25h metro/tranvía). En ese caso, el destino exacto es el `punto_acceso` de esa bajada. Replica la lógica de `crear_mapeos.py` en Spark con `Window.lead()`.  
→ `prediccion_descenso = next_punto_acceso`, `tipo = 'BAJADA'`

**Grupo B — Resto de subidas** (bus, y subidas sin bajada validada):  
Se pasan por `SparkPrediccionDescenso` (trip chaining) y `SparkCheckTransbordo`.  
→ Se conservan solo `tipo = 'BAJADA'` y `tipo = 'TRANSBORDO'`. Los **UNLINKED se descartan** (no tienen destino conocido y no aportan información útil).

El resultado se guarda en `almacen/<nombre>/x_pred_checked_hist_indiv_mdlf.parquet`.

**Prevención de data leakage:**  
El historial completo se carga en `IndividualFeatureExtractor.fit()`, pero en `transform()` se aplica un filtro por viaje: solo se usan filas con `timestamp_validacion < ts_viaje`. Esto garantiza que, si un usuario tiene varios viajes MDLF a predecir, el historial de cada uno no contiene información de los viajes posteriores.

---

#### Extracción de features — `IndividualFeatureExtractor`

**Archivo:** `MDLF/INDIVIDUAL/individual_features.py`

Para cada viaje UNLINKED, extrae el historial de destinos del **propio usuario** desde la misma estación y contexto temporal. Produce cuatro sub-features (sin PROX):

| Sub-feature | Columnas | Sub-vectores | Qué captura |
|---|---|---|---|
| TD | N×2 | `[mismo_daytype+franja, cualquier_daytype+franja]` | Frecuencia de destinos condicionada al contexto temporal del viaje actual |
| SD | N×1 | `[mismo_origen]` | Frecuencia de destinos desde el mismo origen (sin filtro temporal) |
| TSD | N×2 | `[mismo_daytype+franja+origen, cualquier_daytype+franja+origen]` | TD filtrado además por origen |
| TSI | N×1 | `[historial_completo]` | Frecuencia total de destinos (sin ningún filtro) |
| PROX *(opcional)* | N×2 | `[prox_tsi, prox_td]` | Validaciones en otros modos (bus/metro) cercanas a estaciones cercanías |

Los dos sub-vectores de TD y TSD siguen fielmente las ecuaciones (2)(3)(5)(6) del paper (Zhao et al., 2022): el primero filtra por el mismo daytype **y** franja del viaje actual; el segundo filtra solo por franja (todos los días), actuando como suavizado natural cuando el historial es escaso.

El hiperparámetro `td_mode` permite explorar una alternativa no presente en el paper (`'split'`), que sustituye los dos sub-vectores por conteos WD y WE separados, sin condicionar al daytype del viaje actual.

| Concepto | Valor |
|---|---|
| Historial | `x_pred_checked_hist_indiv_mdlf.parquet` (vía `IndividualHistBuilder`), filtrado a usuarios de train ∪ test y con timestamp < ts_viaje |
| Normalización | L2 per-station (fila de la matriz N×6 normalizada independientemente) |
| `td_mode='context'` | Fiel al paper — **default** |
| `td_mode='split'` | Alternativa: columnas WD / WE en lugar de daytype_actual / all_days |
| **Dimensión por estación** | **D_ind = 6** (sin PROX) o **8** (con PROX) — igual en ambos modos |
| **Salida por viaje** | Array aplanado `N × D_ind + 1` (+ flag `has_history`) |

### 4.4 Crowd features (`CrowdFeatureExtractor`)

**Archivo:** `MDLF/CROWD FEATURES/crowd_features.py`

Para cada viaje UNLINKED, estima la distribución de destinos del **colectivo de todos los usuarios** que viajaron desde el mismo origen, mismo daytype y mismo slot temporal, mirando hacia atrás semana a semana. Complementa el historial individual cuando el usuario tiene pocos viajes registrados.

**Fuente de datos:** el mismo `x_pred_checked_hist_indiv_mdlf.parquet` que Individual Features — no tiene fuente propia.

#### `fit()` — pre-agregación del historial colectivo

```
history_df (todos los modos)
    │ filtrar tipo_transporte == 'cercanias'
    │ añadir slot (entero 0-95, paso 15 min) y daytype (WD/WE)
    │ añadir fecha (date sin hora)
    │ descartar filas sin destino (null o 'unknown')
    ▼
groupby([origen, daytype, slot, fecha, destino]) → conteo n
    ▼
índice: dict[(origen, daytype, slot)] → DataFrame(fecha, destino, n)
        ordenado por fecha para búsqueda eficiente por ventana temporal
```

No hay índice por usuario — agrega el comportamiento de **todos** los usuarios desde cada estación.

#### `transform()` — cálculo por viaje UNLINKED

Para cada viaje (origen `s_o`, `daytype`, `slot`, `fecha`):

```
Para m = 1 … n_weeks:
    ventana = [fecha − m×7días, fecha − (m−1)×7días)
    → buscar key=(s_o, daytype, slot) en el índice
    → filtrar registros dentro de la ventana
    → sumar conteos por destino → proporciones (cnt / total)
    → F_c^m ∈ R^N

Apilar [F_c^1, …, F_c^n_weeks] → (N × n_weeks)
    → normalizar L2 por fila (per-station)
    → aplanar → vector N × n_weeks
```

La ventana empieza en `fecha − 1×7d`, nunca incluye el día del viaje → sin data leakage.

#### Diferencias clave respecto a Individual Features

| Aspecto | Individual | Crowd |
|---|---|---|
| **¿Quién?** | Solo el usuario `u` | Todos los usuarios |
| **Granularidad temporal** | `franja` (período funcional) | `slot` exacto (entero de 15 min) |
| **Ventana temporal** | Todo el historial anterior | Ventanas semanales m×7 días |
| **Data leakage** | `ts < ts_viaje` en transform() | Ventana empieza en `fecha − 7d` |
| **Salida** | N×6 (o N×8) | N×n_weeks |

| Concepto | Valor |
|---|---|
| Historial | `x_pred_checked_hist_indiv_mdlf.parquet` (vía `IndividualHistBuilder`), solo cercanías |
| Slot temporal | 15 min (0-95) |
| Normalización | L2 per-station sobre la matriz N×n_weeks |
| **Dimensión por estación** | **D_crowd = n_weeks = 2** (por defecto) |
| **Salida por viaje** | Array aplanado `N × n_weeks + 1` (+ flag `has_crowd_signal`) |

### 4.5 Neighborhood Crowd features — opcional (`ResidenceInferrer` + `NeighborhoodCrowdExtractor`)

**Archivos:** `MDLF/CROWD FEATURES/residence_inferrer.py` y `MDLF/CROWD FEATURES/neighborhood_crowd.py`

**Activación:** `include_neighborhood_crowd=True` en `mdlf_params`.

Extiende la idea de Crowd filtrada al **vecindario de residencia** del viajero, bajo el supuesto de que personas del mismo barrio comparten patrones de movilidad similares.

#### Paso 1 — Inferencia de parada de residencia (`ResidenceInferrer`)

Para cada usuario se identifica el **primer viaje del día** durante todo el historial (cualquier modo de transporte). La estación de la que más frecuentemente parte como primer viaje es la candidata a parada de residencia.

**Criterio de confirmación:**
```
n_días_historial ≥ min_days_residence   (default 5)
  Y
count(estación_top) / count(estación_2ª) ≥ dominance_ratio   (default 2.0)
```

Si el usuario solo arranca siempre desde una misma estación (dominancia absoluta), se confirma directamente. Usuarios que no cumplen los criterios quedan sin residencia asignada → recibirán ceros en esta feature.

#### Paso 2 — Crowd del vecindario (`NeighborhoodCrowdExtractor`)

Una vez conocida la parada de residencia, se filtran los usuarios del historial que comparten esa parada y se agregan sus destinos de cercanías:

```
F_nb^m[s_i] = P(destino=s_i | residencia=r_u, daytype, slot, fecha ∈ [t-m×7d, t-(m-1)×7d))
```

La clave de agrupación es `(residence_station, daytype, slot)` en lugar de `(origen, daytype, slot)`.

**Protección min_neighbors:** si el vecindario tiene menos de `min_neighbors` usuarios (default 10), la señal se descarta y el vector es cero — evita estimaciones extremadamente ruidosas de vecindarios con 2-3 personas.

#### Comportamiento sin señal

Usuarios sin residencia confirmada o con vecindario insuficiente → vector de ceros + flag `has_nbcrowd_signal=0`. La CNN aprende a ignorar esas dimensiones: los pesos Conv1d asociados solo reciben gradientes de los viajes con señal no nula.

| Concepto | Valor |
|---|---|
| Historial para inferencia | Mismos CSVs que Individual (fallback: `x_pred_checked`) |
| Historial para crowd | `x_pred_checked`, solo cercanías |
| Slot temporal | 15 min (igual que crowd global) |
| Normalización | L2 per-station |
| **Dimensión por estación** | **D_nbcrowd = n_weeks = 2** (por defecto, igual que crowd) |
| **Salida por viaje** | Array aplanado `N × n_weeks + 1` (+ flag `has_nbcrowd_signal`) |

---

## 5. Construcción del tensor V

El tensor V representa, para cada viaje, las features de **todos los destinos candidatos** a la vez — entrada de la CNN.

### 5.1 Region Features V_D ∈ R^(N, D_R)

Las Region Features son **estáticas** (no dependen del viaje, solo de la estación). Se construyen concatenando LDA y LINE:

```
V_D = [theta_c ‖ line_c]  ∈ R^(N, D_R)

D_R = K + L = 16 + 64 = 80   (valores por defecto lo de LINE en el paper, lo otro lo cambiamos si quieremos es el num de topicos)
```

`V_D` es la **misma matriz para todos los viajes**. Se usa en el módulo OD Region Correlation para representar las regiones de origen y destino.

### 5.2 Context features — descomposición en temporal y espacial

Las context features del viaje se dividen en dos componentes:

**a) Temporal features (F^t_c):**
- Tipo de día: `daytype ∈ {0=laborable, 1=fin de semana/festivo}`
- Slot temporal: `slot ∈ {0..47}` (intervalos de 30 min, 48 slots/día)
- Se codifican como un **único índice entero**: `idx = daytype × 48 + slot ∈ {0..95}`
- Este índice indexa la tabla de embeddings entrenables E_table → produce **E ∈ R^(D_T)**

**b) Spatial features (v_o):**
- Region attributes de la **estación de origen** del viaje
- `v_o = V_D[station_index.index(s_o)] ∈ R^(D_R)`

Estas dos componentes son los inputs del módulo OD Region Correlation (ver Sección 6).

### 5.3 V_static ∈ R^(n, N, D_static) — features individuales y crowd

Las features individuales y crowd se aplanarán como una fila por destino candidato:

```
V_static[viaje, estación_i, :] = [F_u[estación_i] ‖ F_crowd[estación_i] ‖ F_nbcrowd[estación_i]]

D_static = D_ind + D_crowd + D_nbcrowd

Sin neighborhood crowd (default):  D_static = 6 + 2 + 0 = 8
Con neighborhood crowd activo:     D_static = 6 + 2 + 2 = 10
```

La última columna de cada extractor (flag `has_history` / `has_crowd_signal` / `has_nbcrowd_signal`) se descarta al ensamblar.

### 5.4 Resumen de dimensiones

| Tensor | Shape (default) | Shape (+ neighborhood) | Descripción |
|---|---|---|---|
| `V_D` | `(N, 80)` | `(N, 80)` | Region features de todos los destinos |
| `v_o` | `(n, 80)` | `(n, 80)` | Region feature del origen, por viaje |
| `temporal_idx` | `(n,)` ∈ {0..95} | `(n,)` ∈ {0..95} | Índice temporal por viaje |
| `V_static` | `(n, N, 8)` | `(n, N, 10)` | Ind + crowd (+ neighborhood) por viaje y destino |
| **V (entrada CNN)** | `(n, N, 12)` | `(n, N, 14)` | V_static ‖ F'_R; D = D_static + H |

---

## 6. Módulo OD Region Correlation — atención bilineal

**Sección V.A del paper (Zhao et al., 2022)**

El módulo aprende cómo el **contexto temporal** del viaje modula la correlación entre la región de origen y cada región de destino candidato. Las context features del viaje se descomponen en dos:

### 6.1 Temporal features F^t_c → embedding E (entrenable)

Las temporal features son:
- **daytype**: booleano `{0=laborable, 1=fin de semana/festivo}`
- **slot temporal**: `slot ∈ {0..47}` (intervalos de 30 min, 48 slots/día)

Se codifican como índice único: `idx = daytype × 48 + slot ∈ {0..95}` y se mapean mediante una tabla de embeddings entrenables:

```
E_table ∈ R^(96, D_T)    — nn.Embedding(96, D_T), inicializado aleatoriamente, entrenable
E = E_table[idx] ∈ R^(D_T)    — lookup por viaje, dimensión D_T=16
```

E se aprende por backpropagation junto con el resto del modelo. Captura patrones temporales como "en hora punta de mañana los lunes, ciertos destinos son más probables".

### 6.2 Spatial features — region attributes del origen (v_o)

```
v_o = V_D[station_index.index(s_o)] ∈ R^(D_R)
```

`v_o` son los region attributes (LDA + LINE concatenados) de la estación origen del viaje, extraídos de `V_D`.

### 6.3 Atención bilineal rank-1 multi-head (paper-faithful)

Para cada cabeza `h = 1, …, H`:

```
W^h = W1^h · E · W2^h    ∈ R^(D_R × D_R)
      (D_R×1)(1×D_T)(D_T×D_R)

score_{h,i} = v_o @ W^h @ v_{d_i}^T    (escalar por destino i)
```

La descomposición rank-1 de W^h permite una implementación eficiente sin materializar la matriz D_R×D_R:

```
a_h    = v_o  @ W1^h     →  (batch, 1)     escalar origen
B_h    = V_D  @ W2^h.T   →  (N, D_T)       proyección destinos al espacio temporal
C_h    = E    @ B_h.T    →  (batch, N)      modulación temporal × destino
scores = a_h  * C_h      →  (batch, N)      correlación OD final
```

Resultado:
```
F'_R = stack([scores_1, …, scores_H], axis=2)  ∈ R^(batch, N, H)
```

| Parámetro | Default | Shape | Entrenable |
|---|---|---|---|
| `E_table` | aleatorio | `(96, D_T)` = `(96, 16)` | Sí |
| `W1` | aleatorio | `(H, D_R, 1)` = `(4, 80, 1)` | Sí |
| `W2` | aleatorio | `(H, D_T, D_R)` = `(4, 16, 80)` | Sí |
| `D_R` | 80 | K + L = 16 + 64 | — |
| `D_T` | 16 | dim embedding temporal | — |
| `H = n_heads` | 4 | cabezas de atención | — |

---

## 7. Arquitectura del modelo — `_DestinationCNN`

La CNN toma V ∈ R^(batch, N, D) y produce logits ∈ R^(batch, N), un score por destino candidato.

```
V_static  (batch, N, D_static=8)  ─┐
                                    ├─ concat dim=2 ─→  V (batch, N, D=12)
F'_R      (batch, N, H=4)         ─┘

V → permute(0,2,1) → (batch, D, N)

Conv1d(D→C, kernel=M, padding=(M-1)//2) + ReLU  →  (batch, C, N)
Conv1d(C→C, kernel=M, padding=(M-1)//2) + ReLU  →  (batch, C, N)
Conv1d(C→1, kernel=M, padding=(M-1)//2)          →  (batch, 1, N)

squeeze → logits (batch, N)
```

| Parámetro | Default | Descripción |
|---|---|---|
| `D = D_static + H` | 12 | Dimensión total por estación en CNN |
| `C = cnn_channels` | 16 | Canales internos de la CNN |
| `M = kernel_size` | 1 | Kernel Conv1d sobre N estaciones |
| `N` | variable | Número de estaciones cercanías |

> **Nota sobre M:** Con M=1 cada estación se procesa de forma independiente (equivalente a MLP compartido). Con M=3 se captura contexto local entre estaciones vecinas en el ordenamiento del índice.

**Loss:** `CrossEntropyLoss` — incluye Softmax implícito. La etiqueta es el índice de la estación de destino real en `station_index`.

**Predicción:** `argmax(logits)` para top-1; `topk(logits, k=5)` para top-5.

---

## 8. Entrenamiento

```
Para cada época e = 1, …, n_epochs:
  1. Permutación aleatoria de índices de train
  2. Para cada batch b de tamaño batch_size:
       logits = model(V_static[b], v_o[b], V_D, temporal_idx[b])
       loss   = CrossEntropyLoss(logits, y_true[b])
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
  3. scheduler.step()  (solo si lr_scheduler=True)
  4. Registrar loss medio de la época (y lr actual si scheduler activo)
```

| Parámetro | Default | Notas |
|---|---|---|
| Optimizador | SGD, momentum=0.9 | `optimizer='adamw'` para AdamW |
| `lr` | 0.01 | LR inicial; si `lr_scheduler=True`, es el LR máximo |
| `batch_size` | 64 | |
| `n_epochs` | 20 | Recomendado ≥ 50 si `lr_scheduler=True` |
| `lr_scheduler` | `False` | `True` → `CosineAnnealingLR(T_max=n_epochs, eta_min=lr/100)` |

**Sobre el scheduler coseno:** con LR constante (default), SGD tiende a estancarse a partir de la época 10-15 oscilando alrededor del mínimo. Con `CosineAnnealingLR` el LR empieza alto (exploración) y termina cerca de cero (convergencia fina), lo que normalmente produce una loss final significativamente menor. Se imprime el LR actual en cada época para monitorizar la curva.

---

## 9. Evaluación

Sobre el conjunto de test (20% del labeled set):

| Métrica | Descripción |
|---|---|
| **Top-1 accuracy** | El destino predicho (argmax) coincide con el real |
| **Top-5 accuracy** | El destino real aparece entre los 5 con mayor score |

---

## 10. Atribución de relevancia por bloque de features

**Método:** `PipelineMDLF.attribution_by_group()`, ejecutado sobre el conjunto de test tras el entrenamiento.

### Qué mide

Para cada bloque de features se calcula la **magnitud media del gradiente del logit de la clase correcta** respecto a las entradas de ese bloque. Cuanto mayor es el gradiente, más influye ese bloque en la decisión del modelo.

Formalmente, para cada viaje `i` con destino real `y_i`, se computa:

```
atribución(bloque B) = E_{i,k,d} [ |∂ logit_i[y_i] / ∂ input_B[i,k,d]| ]
```

donde la esperanza es sobre viajes `i`, estaciones candidatas `k`, y dimensiones `d` del bloque.

### Dos passes separados (por arquitectura)

El modelo tiene dos entradas diferenciadas:

**V_static** — features por viaje (Individual, Crowd, Neighborhood):  
Cada viaje `i` tiene su propio tensor `Vs[i, N, D_static]`. Un único forward+backward calcula `∂logit_i[y_i]/∂Vs[i,:,:]` para todos los viajes de forma vectorizada.

```
grad_Vs shape: (n_viajes, N, D_static)
importancia  = mean sobre (n_viajes, N) → vector (D_static,)
```

**V_D** — features de estación compartidas (LDA, LINE):  
`V_D` es una única matriz `(N, D_R)` compartida por todos los viajes. Su gradiente acumula la suma de contribuciones de los `n_viajes`:

```
V_D2.grad[k,d] = Σᵢ ∂logit_i[y_i] / ∂V_D[k,d]     ← suma sobre viajes
importancia    = |V_D2.grad|.mean(axis=N) / n_viajes  ← normalizar a media por viaje
```

La división por `n_viajes` es necesaria para que LDA/LINE sean comparables con Individual/Crowd. Sin ella, la atribución de LDA aparecería inflada por un factor igual al número de viajes de test.

**Bloque TEMPORAL** — embedding E ∈ R^(D_T) del contexto temporal:  
Se captura mediante un forward hook sobre `E_table` que inyecta un tensor hoja con `requires_grad=True`. El gradiente respecto a la salida del embedding (no a los parámetros) se extrae y se promedia sobre viajes y dimensiones.

### Limitación conocida

Para V_D existe una asimetría sutil respecto a V_static: se toma `|Σ grad_i| / n` en lugar de `Σ|grad_i| / n`. Por la desigualdad triangular, `|Σ grad_i| ≤ Σ|grad_i|`, por lo que LDA y LINE podrían estar **ligeramente subestimados** si los gradientes de distintos viajes tienen signos opuestos. En un modelo bien entrenado esta diferencia es pequeña (los gradientes respecto a parámetros compartidos tienden a tener signo consistente), pero conviene tenerla en cuenta al comparar bloques de V_D con bloques de V_static.

### Cómo interpretar los valores

| Situación | Significado |
|---|---|
| LDA >> resto | El modelo predice principalmente por función regional de las estaciones. Frecuente cuando el historial individual es escaso o `norm=None`. |
| Individual/Crowd próximos a LDA | El historial de usuario y el comportamiento colectivo contribuyen de forma apreciable. Requiere historial suficientemente denso y `norm='l2'`. |
| TEMPORAL ≈ 0 | El módulo de correlación OD no está siendo explotado (habitual cuando LDA domina). No indica un error — el embedding temporal es entrenable y su gradiente se anula si la red no lo necesita. |
| Individual > Crowd con norm=None | Artefacto de escala: los raw counts de usuarios frecuentes producen gradientes grandes que compensan artificialmente la normalización. No refleja importancia real. |

### Densidad de señal y por qué LDA suele dominar

LDA y LINE producen features **densas**: todas las N estaciones candidatas tienen valores no nulos. Individual y Crowd son **dispersas**: solo las 2-3 estaciones que el usuario ha visitado previamente tienen valores no nulos; el resto son vectores cero. El CNN promedia sobre las N estaciones, por lo que los gradientes de Individual/Crowd quedan diluidos por las N-k filas de ceros, mientras que LDA no sufre esa dilución. Esto es un reflejo real de la estructura de datos, no un artefacto del método.

---

## 11. Estudio de ablación

El parámetro `ablation` acepta un conjunto de nombres de módulos a desactivar:

```python
ablation={'lda'}           # Sin LDA → K=0, D_R=64, OD module activo
ablation={'lda', 'line'}   # Sin region features → D_R=0, H_eff=0 (OD module OFF)
ablation={'individual'}    # Sin F_u → D_static=D_crowd=n_weeks
ablation={'crowd'}         # Sin F_crowd → D_static=D_ind
```

Efecto en dimensiones:

| Módulo desactivado | D_R | D_static | H_eff | OD module |
|---|---|---|---|---|
| ninguno | 80 | 8 | 4 | ON |
| `lda` | 64 | 8 | 4 | ON |
| `line` | 16 | 8 | 4 | ON |
| `lda` + `line` | 0 | 8 | **0** | **OFF** |
| `individual` | 80 | 2 | 4 | ON |
| `crowd` | 80 | 6 | 4 | ON |
| `individual` + `crowd` | 80 | **0** | 4 | ON |

Cuando `H_eff=0`, el módulo `_ODRegionModule` no se instancia y `F'_R` no se concatena a `V_static`.

---

## 11. Logging de ejecuciones

Cada ejecución de `PipelineMDLF.run()` genera automáticamente un fichero de log en:

```
MDLF/runs/run_YYYYMMDD_HHMMSS.log
```

El log recoge toda la salida estándar y los mensajes de `logging` durante la ejecución: tiempos de cada paso, dimensiones de los tensores, pérdida por época y métricas finales.

---

## 12. Cómo ejecutar

El flujo está separado en dos pasos independientes:

### Paso 1 — Generar el almacén (una sola vez por dataset)

```python
# hipótesis 21 serializa los 5 inputs en MDLF/almacen/<nombre_csv>/
Pipeline(
    '50mil_tarjetas_3_al_5_marzo.csv',
    hipotesis={5, 10, 21},
    hiperparams=hip,
).start()
# → crea MDLF/almacen/50mil_tarjetas_3_al_5_marzo/
```

### Paso 2a — Lanzar MDLF directamente desde notebook (recomendado)

```python
from MDLF.pipeline_mdlf import PipelineMDLF

pipe = PipelineMDLF(
    nombre_almacen='50mil_tarjetas_3_al_5_marzo',
    od_matrices_subdirs=[
        'validaciones_3_7_febrero',
        'validaciones_8_9_febrero',
        'validaciones_10_16_febrero',
        'validaciones_17_23_febrero',
        'validaciones_24febrero_2marzo',
    ],
    # LDA
    n_topics_cercanias=16,        # K → D_R = 16 + 64 = 80
    # Crowd
    n_weeks=2,                    # D_crowd = 2
    # OD Region Correlation
    n_heads=4,                    # H
    d_t=16,                       # D_T (embedding temporal)
    # CNN
    cnn_channels=16,              # C
    kernel_size=1,                # M
    # Entrenamiento
    optimizer='adamw',
    lr=0.001,
    batch_size=64,
    n_epochs=50,
    lr_scheduler=True,            # CosineAnnealingLR (recomendado con n_epochs≥50)
    # Features opcionales
    include_prox=True,            # D_ind = 8 (False → 6)
    include_neighborhood_crowd=True,
    # Otros
    test_size=0.2,
    random_state=42,
    # ttp_paths=[                  # historial individual desde TTP CSVs (primera vez)
    #     '../descomprimido/TTP_PERIODO_01_2025.csv',
    #     ...
    #     '../descomprimido/TTP_PERIODO_06_2025.csv',
    # ],
    # cubo_hiperparams=hip,       # hiperparams del pipeline (dia_virtual, radio_busqueda, ...)
    # ablation={'lda', 'line'}    # descomentar para ablación
).run()

print(f"Top-1: {pipe.test_accuracy:.4f}")
print(f"Top-5: {pipe.test_accuracy5:.4f}")
```

### Paso 2b — Lanzar MDLF desde pipeline de cubo izquierdo (hipótesis 20)

```python
# Requiere que el almacén ya exista (generado con hipótesis 21)
pipe = Pipeline(
    'cualquier_csv.csv',          # no se usa para cargar datos
    hipotesis={20},
    mdlf_params={
        'nombre_almacen': '50mil_tarjetas_3_al_5_marzo',  # obligatorio
        'od_matrices_subdirs': [...],
        ...
    },
)
pipe.start()
print(f"Top-1: {pipe.mdlf.test_accuracy:.4f}")
```

---

## 13. Diagrama de dimensiones completo (valores por defecto)

```
                    ┌─────────────────────────────────────────────────┐
                    │              PIPELINE MDLF                      │
                    │                                                 │
  unknown ──────────┤  _build_labeled_set  ──→  labeled_c (n viajes) │
  x_test / y_test ──┤                                                 │
                    │                                                 │
                    │  LDA  → theta_c  [N × 16]  ─┐                  │
                    │  LINE → line_c   [N × 64]  ─┴→ V_D  [N × 80]  │
                    │                                                 │
                    │  IndividualExtractor → X_ind  [n × (N×6+1)]   │
                    │  CrowdExtractor      → X_crowd [n × (N×2+1)]  │
                    │                                                 │
                    │  _build_V:                                      │
                    │    V_static  [n × N × 8]   = F_u ‖ F_crowd    │
                    │    V_D       [N × 80]                           │
                    │    v_o       [n × 80]       (origen de V_D)    │
                    │    temporal  [n]  ∈ {0..95}                    │
                    │                                                 │
                    │  _DestinationCNN:                               │
                    │    E_table[temporal] → E  [n × 16]  (entrenable)│
                    │    ODRegionModule(v_o, V_D, E) → F'_R [n×N×4] │
                    │    V = V_static ‖ F'_R  →  [n × N × 12]       │
                    │    Conv1d(12→16→16→1)   →  logits  [n × N]    │
                    │    argmax → destino predicho                    │
                    └─────────────────────────────────────────────────┘

Parámetros entrenables:
  E_table : 96 × 16       =  1.536
  W1      : 4 × 80 × 1   =    320   (rank-1, paper-faithful)
  W2      : 4 × 16 × 80  =  5.120
  Conv1   : 12 × 16 × 1  =    192  (+16 bias)
  Conv2   : 16 × 16 × 1  =    256  (+16 bias)
  Conv3   : 16 × 1  × 1  =     16  (+1 bias)
  ──────────────────────────────────
  Total ≈ 7.473 parámetros  (varía con N al no depender de N excepto en V_D)
```
