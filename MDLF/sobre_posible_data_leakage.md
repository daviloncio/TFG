# Data leakage en PipelineMDLF — análisis y conclusión

## Pregunta

Al usar el historial individual (`x_pred_checked_hist_indiv_mdlf.parquet`) para construir las features de `IndividualFeatureExtractor`, ¿se está filtrando información del conjunto de test al de train?

---

## Contexto

El historial cubre todo el período del almacén (p.ej. enero–junio 2025), incluyendo fechas posteriores al `split_date`. `IndividualFeatureExtractor.fit()` recibe este historial completo. La pregunta es si eso introduce leakage.

---

## Conclusión: no hay data leakage

Hay dos mecanismos independientes que lo garantizan.

---

### Mecanismo 1 — los viajes a predecir nunca están en el historial

`x_pred_checked_hist_indiv_mdlf.parquet` solo contiene viajes con `tipo ∈ {BAJADA, TRANSBORDO}`. Los viajes UNLINKED —exactamente los que el modelo debe predecir en train y test— están **excluidos por construcción** en `IndividualHistBuilder.build()`:

```python
pred_trips_pd = pred_trips_pd[pred_trips_pd["tipo"] != "UNLINKED"]
```

Es imposible que el historial contenga un viaje cuyo destino el modelo deba predecir.

---

### Mecanismo 2 — filtro por timestamp por viaje en `transform()`

`IndividualFeatureExtractor.fit()` no calcula estadísticas globales sobre el historial: solo lo **indexa por usuario** en un diccionario. La computación real ocurre en `transform()`, donde para cada viaje UNLINKED se aplica:

```python
# individual_features.py, líneas 252-260
hist_u  = hist_u [hist_u ["timestamp_validacion"] < ts]
other_u = other_u[other_u["timestamp_validacion"] < ts]
```

Esto significa que para un viaje de **test** del 15 de junio, solo se usa historial anterior al 15 de junio. Para un viaje de **train** del 10 de abril, solo historial anterior al 10 de abril. La barrera es por viaje individual, no por split global.

El hecho de que `fit()` haya indexado el historial completo (incluyendo junio) no supone ningún problema: los datos de junio solo serían accesibles para viajes con `ts > fecha_junio`, que no existen en train.

---

### Crowd features — filtro explícito por `split_date`

`CrowdFeatureExtractor` sí recibe un historial ya recortado antes del `fit()`, en `pipeline_mdlf.py`:

```python
# pipeline_mdlf.py, líneas 1049-1053
if self.split_date is not None:
    self._hist_crowd = self.x_pred_checked[
        pd.to_datetime(self.x_pred_checked["timestamp_validacion"]).dt.tz_localize(None)
        < self.split_date
    ]
```

El crowd extractor directamente no ve datos del período de test en ningún caso.

---

## Por qué esto no fue un problema con 50mil pero sí hay que verificarlo con 75mil

Con `50mil_tarjetas_3_al_31_marzo`, el historial también se generó de los 6 TTP CSVs (enero–junio), pero el `split_date` era `2025-03-24` y el historial de los usuarios MDLF tenía muy pocos viajes post-split (última semana de marzo). El solapamiento era mínimo y los mecanismos anteriores lo cubrían igualmente.

Con `75mil_tarjetas_enero_junio_2025` y `split_date = 2025-06-01`, el historial contiene 5 meses antes del split y 1 mes después. El volumen post-split es mayor, lo que hace más importante confirmar que los mecanismos funcionan — y lo hacen.

---

## Punto a vigilar en el futuro

Si en algún momento `IndividualFeatureExtractor.fit()` pasa a calcular **estadísticas globales** sobre el historial completo (medias, distribuciones a priori, normalización global...), habría que filtrar el historial a `< split_date` antes de llamar a `fit()`, igual que se hace ya con Crowd. Por ahora `fit()` solo indexa — no agrega — por lo que el diseño es seguro.
