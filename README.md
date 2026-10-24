# TFG — Inferencia de destinos y reconstrucción de viajes en el transporte público de Madrid

Trabajo de Fin de Grado de David López.

El proyecto reconstruye viajes a partir de las validaciones de entrada con tarjeta de transporte: infiere la parada de bajada, detecta transbordos y construye matrices origen-destino. Para los viajes que quedan sin bajada inferida, predice el destino con una adaptación del modelo multivista MDLF (Zhao et al., 2022).

## Estructura

- `pipeline_inferencia/` — pipeline en pandas y PySpark: preprocesado, predicción de descenso, comprobación de transbordos, reconstrucción de viajes, matrices OD por zonificación y métricas de acierto de bajadas.
- `MDLF/` — implementación del modelo de predicción de destino:
  - `LDA/`: función de cada región/estación a partir de matrices OD.
  - `LINE/`: embeddings de estaciones sobre el grafo de flujos.
  - `INDIVIDUAL/` y `CROWD FEATURES/`: historial del propio usuario y comportamiento colectivo.
  - `pipeline_mdlf.py`: entrenamiento y evaluación del modelo.
  - Notebooks de búsqueda de hiperparámetros (Optuna), ablación y validación temporal.
  - Documentación técnica en `MDLF/DOCUMENTACION.md` y `MDLF/FLUJO_CUBO_MDLF.md`.
- `diagrama_1er_orden.pdf` — diagrama general del sistema.

## Datos

Los datos utilizados (validaciones de tarjeta de transporte y topología de la red) fueron cedidos por el Consorcio Regional de Transportes de Madrid (CRTM) bajo condiciones de confidencialidad. Este repositorio **no incluye esos datos ni ficheros derivados de ellos**, y las salidas de los notebooks se han eliminado por el mismo motivo. Por tanto, el código no puede ejecutarse tal cual sin acceso a esos datos.

## Referencia

Zhao et al. (2022). *A Multi-View-Based Deep Learning Framework for Individual Trip Destination Prediction in Public Transportation Systems.*
