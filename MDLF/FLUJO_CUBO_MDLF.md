# Flujo completo: Cubo Izquierdo → MDLF

Este documento describe de forma lineal y exhaustiva el pipeline completo, desde los datos crudos hasta la predicción de destino para viajes UNLINKED. El objetivo principal es dejar claro **qué produce cada módulo**, **qué archivos de disco lee y escribe**, y **qué recursos del cubo izquierdo son consumidos por MDLF**.

---

## PARTE I — Pipeline del Cubo Izquierdo (`cubo_izquierdo/pipeline.py`)

El fichero `pipeline.py` es el orquestador. Llama a los módulos en orden estricto. Cada paso transforma o enriquece los datos y los guarda en atributos `self.*` para que el siguiente paso los consuma.

---

### Paso 1 — Preprocesado (`preprocesado_cubo_izq.py`)

**Función:** `pipeline_tratamiento(path_csv, hiperparams)`

**Qué hace:**  
Carga el CSV crudo de validaciones y lo limpia: elimina duplicados temporales (mismo modo, Δt ≤ 60s, dist < 200m), establece el día operativo (fecha virtual), elimina tarjetas con más de 20 transacciones/día, elimina validaciones previas a la primera subida del día, filtra pacs L7B anómalos y elimina tarjetas con única subida en el día (sin posibilidad de inferir bajada).

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `{path_csv}` | CSV de transacciones crudas (pasado como argumento) |
| `/home/jupyter-lquijano/david/estudios/topologia/tabla_puntos_acceso_con_coords.csv` | Coordenadas de todos los puntos de acceso |

**Datos que ESCRIBE:** ninguno en disco.

**Produce en memoria:**
- `self.validaciones` → DataFrame con transacciones limpias (todas las filas, incluidas bajadas reales codificadas)
- `self.subidas` → solo las subidas (excluye códigos de descenso)

---

### Paso 2 — Crear Mapeos (`crear_mapeos.py`)

**Función:** `crear_mapeos(validaciones, hiperparams)` o versión Spark `spark_crear_mapeos`

**Qué hace:**  
Construye el **conjunto de validación etiquetado**: recorre las transacciones ordenadas por (tarjeta, tiempo) y, cuando detecta un par (subida → bajada real codificada) verosímil (mismo modo, tiempo ≤ 2h30 para cercanías, ≤ 1h15 para metro/ML, misma fecha operativa), guarda ese par.

**Datos que LEE:** `self.validaciones` (DataFrame en memoria)

**Datos que ESCRIBE:** ninguno.

**Produce en memoria:**
- `self.x_test` → subidas con bajada real conocida (referencia de evaluación)
- `self.y_test` → bajadas reales correspondientes a `x_test`

---

### Paso 3 — Predicción de Descenso (`prediccion_descenso/spark_PrediccionDescenso.py`)

**Clase:** `SparkPrediccionDescenso`

**Qué hace:**  
Para cada subida en `self.subidas`, predice la estación de bajada buscando paradas candidatas dentro de un radio de 650m de la siguiente subida del usuario, filtrando por modo de transporte, línea y sentido de ruta.

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `/topologia/paradas_vecinas_1500m_OSM.csv` | Vecinas de cada parada en radio 1500m (grafo OSM) |
| `/RESCATE_csv_bien/rutas_interurbano_urbano.csv` | Rutas de bus interurbano/urbano |
| `/RESCATE_csv_bien/pac_interurbanos.csv` | Puntos de acceso interurbanos |
| `/RESCATE_csv_bien/rutas_EMT.csv` | Rutas EMT |
| `/RESCATE_csv_bien/topologia_solo_EMT.csv` | Topología solo EMT |
| `/RESCATE_csv_bien/rutas_ml.csv` | Rutas metro ligero |
| `/RESCATE_csv_bien/pacs_ml.csv` | Puntos de acceso metro ligero |
| `/RESCATE_csv_bien/topologia_trenes2.csv` | Topología ferroviaria (cercanías) |
| `/cubo_izquierdo/correspondencias_lineas.csv` | Tabla de correspondencias entre líneas |

**Datos que ESCRIBE (solo si hipótesis 'lee'):** temporales en `.../prediccion_descenso/tmp_lote_*` (parquet).

**Produce en memoria:**
- `self.x_1sub` → DataFrame con cada subida + columna `prediccion_descenso` (punto_acceso predicho, o 'unknown')
- `self.x_2sub` → segunda subida de cada par (contexto)
- `self.unknown` → subidas cuya bajada no pudo predecirse → **estos son los viajes UNLINKED que MDLF intentará resolver**

---

### Paso 4 — Check Transbordo (`check_transbordo/check_transbordo.py`)

**Clase:** `SparkCheckTransbordo`

**Qué hace:**  
Reclasifica cada validación de `self.x_1sub` como `BAJADA`, `TRANSBORDO` o `UNLINKED`:
- Si `prediccion_descenso == 'unknown'` → `UNLINKED`
- Si el tiempo entre subida y siguiente subida cabe dentro del tiempo de viaje estimado + 18 min de margen de transbordo → `TRANSBORDO`
- En otro caso → `BAJADA`

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `/cubo_izquierdo/check_transbordo/buffer_claude.csv` | Tiempos de viaje precalculados entre pares (origen, destino) |

**Datos que ESCRIBE:** ninguno.

**Produce en memoria:**
- `self.x_pred_checked` → `x_1sub` enriquecido con columna `tipo` (BAJADA / TRANSBORDO / UNLINKED)

---

### Paso 5 — Transbordos Implícitos (`transbordos_implicitos.py`)

**Qué hace:**  
Detecta cadenas A→B→C→D donde B y C son etapas intermedias sin bajada explícita. Asigna la bajada de D como destino de la cadena completa.

> **Nota de diseño:** Los casos A→B→C→D con más de un transbordo implícito quedan con `null`; se aceptó esta limitación porque son muy pocos. No se propone solución salvo que se pida explícitamente.

**Produce en memoria:**
- `self.transbordos_implicitos` → DataFrame con transbordos implícitos detectados y asignados

---

### Paso 6 — Matrices OD (`crear_matrices.py`)

**Función:** `crear_matriz_od(x_pred_checked, out_dir, tabla_paradas_path)`

**Qué hace:**  
A partir de `x_pred_checked` (todos los viajes con tipo asignado), reconstruye los viajes completos y agrega los flujos origen→destino en intervalos de **15 minutos**, distinguiendo laboral (`WD`) de fin de semana (`WE`).

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `self.x_pred_checked` | DataFrame en memoria (todos los viajes) |
| `/topologia/tabla_puntos_acceso_con_coords.csv` | Nombres y coordenadas de paradas |

**Datos que ESCRIBE:**
| Archivo | Descripción |
|---|---|
| `cubo_izquierdo/matrices/{timestamp}/matriz_od.csv` | Matriz OD en formato long: (intervalo, origen, destino, num_viajes) |

**Produce en disco:**  
→ **`matriz_od.csv`** — este fichero es la entrada principal del módulo LDA de MDLF.

---

### Paso 7 — Accuracy (`accuracy_bajadas.py`)

**Clase:** `Accuracy`

**Qué hace:**  
Evalúa la precisión de las predicciones comparando `x_pred_checked` con el conjunto `x_test / y_test`. Calcula accuracy exacto (misma parada) y aproximado (parada vecina).

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `./near/near_stations.csv` | Paradas vecinas para todos los modos |
| `./near/near_stations_cercanias.csv` | Paradas vecinas cercanías |
| `./near/near_stations_metro.csv` | Paradas vecinas metro |
| `./near/near_stations_ml.csv` | Paradas vecinas metro ligero |

**Produce en memoria:**  
Métricas de accuracy exacto y aproximado por modo.

---

### Paso 8a — Guardar inputs MDLF en almacén (`pipeline.py`, hipótesis 21)

Cuando `21 in hipotesis`, el pipeline serializa los 5 DataFrames necesarios para MDLF en:

```
MDLF/almacen/<nombre_crudo>/
    x_pred_checked.parquet
    unknown.parquet
    x_test.parquet
    y_test.parquet
    subidas_sin_pred.parquet   ← vacío si no aplica (modo munizaga)
```

El `<nombre_crudo>` se deriva automáticamente con `Path(self.path).stem`. Por ejemplo, si el CSV de entrada es `50mil_tarjetas_3_al_5_marzo.csv`, la carpeta será `MDLF/almacen/50mil_tarjetas_3_al_5_marzo/`.

**Esto solo hay que ejecutarlo una vez.** A partir de aquí, MDLF puede lanzarse directamente sin re-ejecutar el pipeline de cubo izquierdo.

---

### Paso 8b — Activación de MDLF (`pipeline.py`, hipótesis 20)

Cuando `20 in hipotesis`, `pipeline.py` instancia `PipelineMDLF` con el nombre del almacén (obligatorio vía `mdlf_params`):

```python
PipelineMDLF(
    nombre_almacen = "50mil_tarjetas_3_al_5_marzo",  # subcarpeta en MDLF/almacen/
    od_matrices_subdirs = [...],
    **resto_de_mdlf_params,
)
```

`PipelineMDLF` carga los 5 parquets desde `MDLF/almacen/<nombre_almacen>/` en su `__init__`. Si la carpeta no existe, lanza `FileNotFoundError` indicando que hay que ejecutar primero la hipótesis 21.

También es posible llamar a `PipelineMDLF` directamente desde un notebook, sin pasar por `Pipeline` de cubo izquierdo:

```python
from MDLF.pipeline_mdlf import PipelineMDLF

pipe = PipelineMDLF(
    nombre_almacen="50mil_tarjetas_3_al_5_marzo",
    ...
).run()
```

---

## PARTE II — Pipeline MDLF (`MDLF/pipeline_mdlf.py`)

MDLF resuelve el subproblema de los viajes `UNLINKED` de cercanías: aquellos para los que el pipeline principal no pudo predecir una bajada. El modelo combina cuatro vistas del comportamiento del viajero y las procesa con una CNN 1D.

---

### Recursos heredados de Cubo Izquierdo

Los 5 inputs de cubo izquierdo se consumen **desde disco** (almacén generado por hipótesis 21). No se pasan en memoria:

| Fichero en almacén | Origen en cubo izquierdo | Uso en MDLF |
|---|---|---|
| `almacen/<nombre>/x_pred_checked.parquet` | Paso 4 (CheckTransbordo) | Historial para crowd e individual; filtrar UNLINKED |
| `almacen/<nombre>/unknown.parquet` | Paso 3 (PredicciónDescenso) | Candidatos UNLINKED a predecir |
| `almacen/<nombre>/x_test.parquet` | Paso 2 (CrearMapeos) | Intersecar con UNLINKED para obtener etiquetas reales |
| `almacen/<nombre>/y_test.parquet` | Paso 2 (CrearMapeos) | Etiquetas reales de los UNLINKED con bajada conocida |
| `almacen/<nombre>/subidas_sin_pred.parquet` | Paso 3 (solo Munizaga) | Ampliar labeled set |
| `cubo_izquierdo/matrices/*/matriz_od.csv` | Paso 6 (Matrices OD) | Entrada al módulo LDA para representaciones de región |
| Grafo NetworkX de cercanías | `check_transbordo/` (pkl) | Entrada al módulo LINE para embeddings topológicos |

---

### Paso M1 — Construir conjunto etiquetado (`_build_labeled_set`)

**Qué hace:**  
Filtra `x_pred_checked` por `tipo == 'UNLINKED'` y red cercanías (prefijo `'04'`). Luego interseca con `x_test / y_test` para quedarse solo con UNLINKED que tienen bajada real conocida (para poder evaluar el modelo).

**Datos que LEE:**
- `self.x_pred_checked` — cargado desde `almacen/<nombre>/x_pred_checked.parquet`
- `self.unknown` — cargado desde `almacen/<nombre>/unknown.parquet`
- `self.x_test`, `self.y_test` — cargados desde `almacen/<nombre>/x_test.parquet` y `y_test.parquet`

**Produce en memoria:**
- `self.labeled_c` → DataFrame de UNLINKED cercanías con columna `destino_real`

---

### Paso M2 — Split train/test (`_split`)

**Qué hace:**  
Divide `self.labeled_c` en entrenamiento y test estratificado por destino.

**Produce en memoria:**
- `self.train_c` → conjunto de entrenamiento
- `self.test_c` → conjunto de test

---

### Paso M3 — Módulo LDA: Region Function (`MDLF/LDA/region_function_lda.py`)

**Clase:** `RegionFunctionLDA`

**Propósito:**  
Aprender una representación funcional de cada estación de cercanías basada en sus patrones de flujo. La idea es que dos estaciones son funcionalmente similares si tienen distribuciones parecidas de destinos en determinadas franjas horarias (residencial, comercial, universitaria, etc.).

**Metáfora:** cada estación es un "documento", y cada par (destino, franja) es una "palabra". LDA extrae K tópicos que representan funciones urbanas.

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `cubo_izquierdo/matrices/*/matriz_od.csv` | Ficheros de matrices OD (uno por día o semana). Columnas: `intervalo`, `origen_punto_acceso`, `destino_punto_acceso`, `num_viajes` |

Se cargan **todos los CSV disponibles** en el directorio de matrices y se concatenan antes de ajustar el modelo.

**Datos que ESCRIBE:**
| Archivo | Descripción |
|---|---|
| `MDLF/LDA/output/theta_cercanias.parquet` | Distribuciones de tópicos para cercanías [N_estaciones × K] |
| `MDLF/LDA/output/theta_metro.parquet` | Distribuciones de tópicos para metro [N_estaciones × K] |
| `MDLF/LDA/output/topics_cercanias.csv` | Descripción de cada tópico (palabras más relevantes) |
| `MDLF/LDA/output/topics_metro.csv` | Idem para metro |

**Produce en memoria:**
- `self.theta_` → `pd.DataFrame` [N_estaciones × K] — distribución de tópicos por estación
- `self.station_index_` → lista de N estaciones

**Parámetros relevantes:**
- `n_topics`: K = 32 para metro, K = 16 para cercanías
- `filter_prefixes`: `['04']` para cercanías, `['05', '06']` para metro
- Tokens: `OUT_{destino}_{slot}_{WD|WE}` y `IN_{origen}_{slot}_{WD|WE}`
- Realmente dejamos metro atras, no va a sa.ir un buen entram¡namiento

---

### Paso M4 — Módulo LINE: Station Embeddings (`MDLF/LINE/station_line.py`)

**Clase:** `StationLINE`

**Propósito:**  
Aprender embeddings topológicos de las estaciones: dos estaciones que están directamente conectadas (first-order) o que comparten muchos vecinos (second-order) tendrán embeddings similares. Esto captura la posición de cada estación dentro de la red de transporte.

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| Grafo NetworkX de cercanías | Construido a partir de los pkl en `cubo_izquierdo/check_transbordo/`. Nodos = estaciones, aristas = tiempos de viaje entre estaciones conectadas |

El grafo es el mismo que usa `check_transbordo` para calcular tiempos de viaje.

**Datos que ESCRIBE:**
| Archivo | Descripción |
|---|---|
| `MDLF/LINE/output/line_cercanias.parquet` | Embeddings cercanías a nivel estación [N × 64] |
| `MDLF/LINE/output/line_cercanias_ap.parquet` | Embeddings cercanías a nivel punto_acceso [N_ap × 64] |
| `MDLF/LINE/output/line_metro.parquet` | Embeddings metro a nivel estación [N × 64] |
| `MDLF/LINE/output/line_metro_ap.parquet` | Embeddings metro a nivel punto_acceso [N_ap × 64] |

**Produce en memoria:**
- `self.embeddings_` → `pd.DataFrame` [N × 64]: 32 dims first-order + 32 dims second-order

**Parámetros relevantes:**
- `dim = 32` → embedding final de 64 dims (FO + SO concatenados)
- `num_epochs_fo = 200`, `num_epochs_so = 1000`
- Peso de arista = `1 / tiempo_viaje` (similitud inversa al tiempo)

---

### Paso M5 — Módulo Individual (`MDLF/INDIVIDUAL/`)

#### Construcción del historial: `IndividualHistBuilder`

**Archivo:** `MDLF/INDIVIDUAL/hist_builder.py`

Antes de extraer features, se necesita un historial rico de viajes procesados para cada usuario. `IndividualHistBuilder` construye y cachea ese historial desde los TTP CSVs históricos (~81 GB).

**Prioridad de carga (la primera que exista gana):**
1. Caché en `almacen/<nombre>/x_pred_checked_hist_indiv_mdlf.parquet` → carga directa sin Spark
2. Si no hay caché y se proporcionan `ttp_paths` → se construye (`build()`)
3. Si no hay ni caché ni `ttp_paths` → fallback a `x_pred_checked` del almacén (solo período actual, con aviso)

**Flujo de `build()`:**

```
TTP CSVs (6 períodos) → filtro id_tarjeta (solo usuarios MDLF)
                      → filtro fecha ≤ ts_max (timestamp máx. del conjunto MDLF)
                      → join coords (tipo_transporte, LATITUD, LONGITUD)
                      → día operativo
                      → deduplicación Haversine (misma tarjeta, mismo modo, ≤60s, <200m)
                      → Window lead() sobre todas las validaciones del día:
                            next_id_resultado, next_punto_acceso,
                            next_tipo_transporte, next_timestamp
                      → filtrar a subidas
                      │
                      ├── tiene_bajada_real? (replicando lógica de crear_mapeos.py)
                      │     · next_id_resultado ∈ códigos_descenso
                      │     · mismo tipo_transporte (metro/cercanías/tranvía)
                      │     · tiempo verosímil (cercanías ≤2.5h, metro ≤1.25h)
                      │
                      ├── SÍ → prediccion_descenso = next_punto_acceso, tipo='BAJADA'
                      │         (destino exacto sin necesidad de trip chaining)
                      │
                      └── NO → SparkPrediccionDescenso (trip chaining)
                                → SparkCheckTransbordo
                                → descartar tipo='UNLINKED' (sin destino conocido)
                                → BAJADA + TRANSBORDO

→ concat ambos grupos → x_pred_checked_hist_indiv_mdlf.parquet
```

**Racionalidad de la estrategia híbrida:**  
En metro y cercanías el usuario valida al salir (torniquete de bajada). Cuando la siguiente validación tras una subida es un código de descenso del mismo modo y en tiempo verosímil, conocemos el destino exacto — más preciso que trip chaining. Para bus (sin torniquete de salida) y resto de casos, se usa trip chaining como siempre.

Los viajes **UNLINKED** se excluyen del historial: no tienen destino conocido y no aportan información útil a las features individuales.

**Prevención de data leakage:**  
El filtro fino de timestamp (`timestamp_validacion < ts_viaje`) se aplica en `IndividualFeatureExtractor.transform()`, por viaje, para garantizar que el historial usado en cada predicción solo contiene viajes anteriores a ese viaje concreto.

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `almacen/<nombre>/x_pred_checked_hist_indiv_mdlf.parquet` | Caché del historial procesado (si existe) |
| `../descomprimido/TTP_PERIODO_0X_2025.csv` (×6) | CSVs TTP crudos si no hay caché |
| `/topologia/tabla_puntos_acceso_con_coords.csv` | Coords y tipo_transporte por punto_acceso |

**Datos que ESCRIBE:**
| Archivo | Descripción |
|---|---|
| `almacen/<nombre>/x_pred_checked_hist_indiv_mdlf.parquet` | Historial procesado (cacheado para ejecuciones posteriores) |

---

#### Extracción de features: `IndividualFeatureExtractor`

**Archivo:** `MDLF/INDIVIDUAL/individual_features.py`

**Propósito:**  
Extraer features del historial personal de cada usuario. Responde a: ¿a dónde suele ir este usuario desde esta estación a esta hora?

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `x_pred_checked_hist_indiv_mdlf.parquet` | Historial procesado (construido por `IndividualHistBuilder`) |
| `/topologia/paradas_vecinas_1500m.csv` | Para la feature PROX: validaciones cercanas en otros modos |

**Datos que ESCRIBE:** ninguno.

**Produce en memoria:**
- Para cada viaje UNLINKED: vector de features [N_estaciones × D_ind] donde D_ind = 6 features:

| Feature | Dimensión | Descripción |
|---|---|---|
| **TD** (Time-Dependent) | 2 | Frecuencia de destinos en mismo slot + tipo_día (laboral/finde) |
| **SD** (Space-Dependent) | 1 | Frecuencia de destinos desde mismo origen (sin filtro temporal) |
| **TSD** (Time-Space-Dep.) | 2 | SD filtrado por slot + tipo_día |
| **TSI** (Time-Series-Indep.) | 1 | Frecuencia total de destinos (sin filtro) |

> Opcionalmente **PROX** añade 2 dimensiones más: validaciones cercanas en bus/metro.

- `self.X_ind_train` → `pd.DataFrame` [n_train × (N × D_ind)]
- `self.X_ind_test` → `pd.DataFrame` [n_test × (N × D_ind)]

---

### Paso M6 — Módulo Crowd (`MDLF/CROWD FEATURES/crowd_features.py`)

**Clase:** `CrowdFeatureExtractor`

**Propósito:**  
Extraer features de comportamiento colectivo: ¿a dónde va la gente en general desde esta estación a esta hora? Se basa en el historial de **todos** los usuarios, no solo del viajero en cuestión.

**Datos que LEE:**
| Archivo | Descripción |
|---|---|
| `almacen/<nombre>/x_pred_checked.parquet` | Historial de viajes de todos los usuarios. Se filtra por `red == '04'` (cercanías) y por `tipo == 'BAJADA'` (bajadas confirmadas, para construir el histórico) |

**Datos que ESCRIBE:** ninguno.

**Produce en memoria:**
- Para cada viaje UNLINKED: vector de frecuencias colectivas [N_estaciones × n_weeks]:
  - `n_weeks = 2` semanas atrás (configurable)
  - Filtrado por slot horario + tipo_día
  - Normalizado L2 por estación

- `self.X_crowd_train` → `pd.DataFrame` [n_train × (N × n_weeks)]
- `self.X_crowd_test` → `pd.DataFrame` [n_test × (N × n_weeks)]

**Clase adicional:** `NeighborhoodCrowdExtractor` (opcional)  
- Infiere la parada de residencia de cada usuario via `ResidenceInferrer`
- Agrega crowd features solo de usuarios que residen en el mismo vecindario
- Útil para usuarios con poco historial individual
- **Datos que LEE:** mismo `almacen/<nombre>/x_pred_checked.parquet`

---

### Paso M7 — Ensamblado del tensor V (`_build_V`)

**Qué hace:**  
Combina todas las features anteriores en los tensores que necesita la CNN.

**No lee ningún archivo de disco.** Usa lo que ya está en memoria:

| Tensor | Forma | De dónde viene |
|---|---|---|
| `V_static` | [n, N, D_static] | Concatenación de features Individual + Crowd por destino |
| `V_D` | [N, D_R] | LDA theta (K dims) + LINE embeddings (64 dims) = 80 dims por destino |
| `v_o` | [n, D_R] | LDA theta + LINE embeddings de la **estación origen** |
| `temporal_idx` | [n] | Índice de slot horario × tipo_día (para embedding temporal) |

Donde:
- **n** = número de viajes en el batch
- **N** = número de estaciones candidatas (≈ 40-60 en cercanías)
- **D_static** = D_ind + n_weeks = 6 + 2 = 8 (sin opcionales)
- **D_R** = K + 64 = 16 + 64 = 80 (region features)

---

### Paso M8 — Red CNN (`MDLF/pipeline_mdlf.py`, clase `_DestinationCNN`)

**Arquitectura:**

```
Entrada:
  V_static   [batch, N, D_static]   ← features individuales + crowd por destino
  v_o        [batch, D_R]            ← región de la estación origen
  V_D        [N, D_R]                ← región de cada estación destino
  temporal   [batch]                 ← índice de franja horaria

Módulo ODRegionCorrelation:
  → Atención bilineal multi-head(v_o, V_D, temporal) = F'_R  [batch, N, H]
  → Captura correlación entre la función de la estación origen y cada destino,
    modulada por el contexto temporal

Concatenación:
  V = [V_static ‖ F'_R]             [batch, N, D_static + H]

CNN 1D (sobre dimensión N):
  Conv1d(D_total → C, kernel M) + ReLU
  Conv1d(C → C, kernel M) + ReLU
  Conv1d(C → 1, kernel M)

Salida:
  logits   [batch, N]   → softmax → probabilidad de cada destino
```

**Entrenamiento:**
- Loss: CrossEntropyLoss
- Optimizador: SGD (momentum 0.9) o AdamW
- Epochs: 20, batches: 64

**Produce:**
- `self.model` → red entrenada
- `self.test_accuracy` → top-1 accuracy en test
- `self.test_accuracy5` → top-5 accuracy en test

---

## PARTE III — Diagrama de flujo de datos (resumen)

```
DATOS CRUDOS (CSV validaciones)
        │
        ▼
[preprocesado_cubo_izq.py]
  self.validaciones (df limpio)
  self.subidas
        │
        ▼
[crear_mapeos.py]
  self.x_test  ←── referencia de evaluación (bajadas reales conocidas)
  self.y_test  ←──────────────────────────────────────────────────────────┐
        │                                                                  │
        ▼                                                                  │
[spark_PrediccionDescenso.py]                                              │
  self.x_1sub  (prediccion_descenso por subida)                           │
  self.unknown (sin predicción → futuros UNLINKED)                        │
        │                                                                  │
        ▼                                                                  │
[spark_check_transbordo.py]                                               │
  self.x_pred_checked  (tipo: BAJADA / TRANSBORDO / UNLINKED)             │
        │                                                                  │
        ├──────────────────────────────────────────────────────────────────┘
        │           ↓ (todo x_pred_checked)
        │     [crear_matrices.py]
        │       cubo_izquierdo/matrices/*/matriz_od.csv  ←── ESCRIBE EN DISCO
        │
        ▼
[if 21 in hipotesis] → guarda 5 parquets en MDLF/almacen/<nombre_crudo>/
        │                    (x_pred_checked, unknown, x_test, y_test, subidas_sin_pred)
        │
        ▼
[if 20 in hipotesis] → PipelineMDLF(nombre_almacen="50mil_tarjetas_3_al_5_marzo")
        │               carga los 5 parquets desde MDLF/almacen/<nombre_almacen>/
        │
        ├── [LDA]  lee  matrices/*/matriz_od.csv ─────────────────── theta_  [N×K]
        │                                                                │
        ├── [LINE] lee  grafo NetworkX (pkl check_transbordo) ───── embeddings_ [N×64]
        │                                                                │
        ├── [IndividualHistBuilder] caché? → carga hist_indiv_mdlf.parquet
        │     si no → TTP CSVs → preprocesado → SparkPrediccionDescenso
        │             + SparkCheckTransbordo → filtra UNLINKED → parquet
        │
        ├── [Individual] lee hist_indiv_mdlf (por tarjeta, filtro ts < ts_viaje) X_ind [n×N×6]
        │                                                                │
        ├── [Crowd]      lee x_pred_checked (historial colectivo) ── X_crowd [n×N×2]
        │                                                                │
        └── [CNN] combina todo:                                          │
              V_D     = theta_ ‖ embeddings_  [N × 80]  ◄──────────────┤
              v_o     = theta_[origen] ‖ emb[origen]    [n × 80]       │
              V_static = X_ind ‖ X_crowd                [n × N × 8]    │
              F'_R    = atención(v_o, V_D, temporal)    [n × N × H]    │
              logits  = CNN(V_static ‖ F'_R)            [n × N]        │
                                                                         │
              → top-1 / top-5 accuracy ◄──── comparado con y_test ◄────┘
```

---

## PARTE IV — Tabla de ficheros de disco por módulo

Esta tabla responde a: **¿qué archivo de disco carga exactamente cada módulo MDLF?**

| Módulo MDLF | Archivo leído | Cuándo se genera ese archivo |
|---|---|---|
| **LDA** | `cubo_izquierdo/matrices/*/matriz_od.csv` | Paso 6 del cubo izquierdo (`crear_matrices.py`) |
| **LINE** | Grafo NetworkX (pkl de `cubo_izquierdo/check_transbordo/`) | Se construye en check_transbordo; se serializa al instanciar `SparkCheckTransbordo` |
| **IndividualHistBuilder** | `MDLF/almacen/<nombre>/x_pred_checked_hist_indiv_mdlf.parquet` (caché) | Generado por `hist_builder.py` la primera vez (o cargado si ya existe) |
| **IndividualHistBuilder** (si no hay caché) | `../descomprimido/TTP_PERIODO_0X_2025.csv` (×6) | CSVs TTP crudos de todos los períodos |
| **Individual** | `x_pred_checked_hist_indiv_mdlf.parquet` (vía `IndividualHistBuilder`) | `hist_builder.py` |
| **Individual (PROX)** | `/topologia/paradas_vecinas_1500m.csv` | Archivo estático de topología |
| **Crowd** | `MDLF/almacen/<nombre>/x_pred_checked.parquet` (cargado en `__init__`) | Hipótesis 21 del cubo izquierdo |
| **Crowd (residencia)** | `x_pred_checked_hist_indiv_mdlf.parquet` (vía `IndividualHistBuilder`) | `hist_builder.py` |
| **CNN** | Nada de disco — opera sobre tensores en memoria | Pasos M3-M6 (LDA, LINE, Individual, Crowd) |

---

## PARTE V — Qué sale de cada módulo MDLF (salidas a disco)

| Módulo | Archivo escrito | Contenido |
|---|---|---|
| **LDA** | `MDLF/LDA/output/theta_cercanias.parquet` | Distribuciones tópicos cercanías [N×16] |
| **LDA** | `MDLF/LDA/output/theta_metro.parquet` | Distribuciones tópicos metro [N×32] |
| **LDA** | `MDLF/LDA/output/topics_cercanias.csv` | Top palabras por tópico (cercanías) |
| **LDA** | `MDLF/LDA/output/topics_metro.csv` | Top palabras por tópico (metro) |
| **LINE** | `MDLF/LINE/output/line_cercanias.parquet` | Embeddings estación cercanías [N×64] |
| **LINE** | `MDLF/LINE/output/line_cercanias_ap.parquet` | Embeddings punto_acceso cercanías [N_ap×64] |
| **LINE** | `MDLF/LINE/output/line_metro.parquet` | Embeddings estación metro [N×64] |
| **LINE** | `MDLF/LINE/output/line_metro_ap.parquet` | Embeddings punto_acceso metro [N_ap×64] |
| **Pipeline MDLF** | `MDLF/runs/run_{timestamp}.log` | Log completo de ejecución (loss, accuracy por época) |
