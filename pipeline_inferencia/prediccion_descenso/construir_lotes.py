from pyspark.sql import functions as F
from pyspark.sql.window import Window
import ipdb
def construir_lotes(self) -> None:
    
    """aqui tenemos que trabajar distinto a lo normal pq spark no trabaja con indices igual q en pandas,
    identifico con una col adicional la primera transaccion de cada dia de cada tarjeta con num_viaje = 1
    
    usamos pair_id porque en spark no tenemos la correspondencia fila por fila de pandas

    la segunda sub es la subida sucesora de una subida a la que le estamos buscanso descenso,
    el punto de acceso de la 2a sub es en el que pondremos nuestro radio para localizar a las candidatas.
    y cada pair_id le vamos a asignar su 2a subida, si la subida es la ultima del dia asignamos como
    2a sub la primera subida del dia"""

    
    w_dia = Window.partitionBy("id_tarjeta","fecha").orderBy(
        F.col("timestamp_validacion").asc(),
        F.col("punto_acceso").asc()
    )

    self.lote_primera_sub = (
        self.lote
        .withColumn("num_viaje", F.row_number().over(w_dia))
        .withColumn("pair_id", F.xxhash64("id_tarjeta", "fecha", "num_viaje"))
        .cache()
    )
    self.lote_primera_sub.count()

    w_dia_full = w_dia.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    
    cols = self.lote_primera_sub.columns
    exprs_segunda = []
    for c in cols:
        if c == 'pair_id':
            exprs_segunda.append(F.col("pair_id"))
            
        else:
            siguiente = F.lead(F.col(c), 1).over(w_dia)
            primera = F.first(F.col(c), ignorenulls=True).over(w_dia_full)
            exprs_segunda.append(F.coalesce(siguiente, primera).alias(c))
           

    self.lote_segunda_sub = self.lote_primera_sub.select(*exprs_segunda).cache()
    self.lote_segunda_sub.count()

   

    sub1_ids = self.lote_primera_sub.select("pair_id").distinct()
    seg2_ids = self.lote_segunda_sub.select("pair_id").distinct()
    

    

    missing_in_sub1 = seg2_ids.join(sub1_ids, "pair_id", "left_anti").count()

    self.lote_segunda_sub = (
        self.lote_segunda_sub
        .withColumn('tipo_asignacion_2sub',F.lit("normal"))
    )

def construir_lotes_next_day(self) -> None:
    """
    Igual que el método base, pero si una subida es la única del día,
    intentamos usar como segunda subida la primera validación del día siguiente exacto.
    
    Si no existe validación en ese día siguiente, esa subida pasa a self.df_unico_dia.
    """

    w_dia = Window.partitionBy("id_tarjeta", "fecha").orderBy(
        F.col("timestamp_validacion").asc(),
        F.col("punto_acceso").asc()
    )

    self.lote_primera_sub = (
        self.lote_inicial
        .withColumn("num_viaje", F.row_number().over(w_dia))
        .withColumn("pair_id", F.xxhash64("id_tarjeta", "fecha", "num_viaje"))
        .withColumn(
            "n_validaciones_dia",
            F.count("*").over(Window.partitionBy("id_tarjeta", "fecha"))
        )
        .cache()
    )
    self.lote_primera_sub.count()

    w_dia_full = w_dia.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    primeras_del_dia = (
        self.lote_primera_sub
        .filter(F.col("num_viaje") == 1)
    )

    cols = self.lote_primera_sub.columns

    primeras_del_dia_ren = primeras_del_dia.select(
        F.col("id_tarjeta").alias("id_tarjeta_next"),
        F.col("fecha").alias("fecha_next"),
        *[
            F.col(c).alias(f"nextday_{c}")
            for c in cols
        ]
    )

    base = (
        self.lote_primera_sub
        .withColumn("fecha_siguiente_esperada", F.date_add(F.col("fecha"), 1))
        .join(
            primeras_del_dia_ren,
            on=(
                (F.col("id_tarjeta") == F.col("id_tarjeta_next")) &
                (F.col("fecha_siguiente_esperada") == F.col("fecha_next"))
            ),
            how="left"
        )
        .drop("id_tarjeta_next", "fecha_next")
        .cache()
    )
    base.count()

    self.df_impredecible_next_day = (
        base
        .filter(
            (F.col("n_validaciones_dia") == 1) &
            (F.col("nextday_pair_id").isNull())
        )
        .select(*self.lote_primera_sub.columns)
        .cache()
    )
    self.df_impredecible_next_day.count()

    exprs_segunda = []

    for c in cols:
        if c == "pair_id":
            exprs_segunda.append(F.col("pair_id"))
        else:
            siguiente = F.lead(F.col(c), 1).over(w_dia)
            primera_mismo_dia = F.first(F.col(c), ignorenulls=True).over(w_dia_full)
            primera_next_day = F.col(f"nextday_{c}")
    
            exprs_segunda.append(
                F.when(
                    F.col("n_validaciones_dia") == 1,
                    primera_next_day
                ).otherwise(
                    F.coalesce(siguiente, primera_mismo_dia)
                ).alias(c)
            )

    exprs_segunda.append(
    F.when(
        (F.col("n_validaciones_dia") == 1) & F.col("nextday_pair_id").isNotNull(),
        F.lit("next_day")
    ).otherwise(
        F.lit("normal")
    ).alias("tipo_asignacion_2sub")
    )
    
    ids_impredecible_next_day  = self.df_impredecible_next_day.select("pair_id").distinct()
    

    self.lote_primera_sub = (
        self.lote_primera_sub
        .join(ids_impredecible_next_day , on="pair_id", how="left_anti")
        .cache()
    )
    self.lote_primera_sub.count()
    

    self.lote_segunda_sub = (
        base
        .select(*exprs_segunda)
        .join(ids_impredecible_next_day , on="pair_id", how="left_anti")
        .cache()
    )
    self.lote_segunda_sub.count()
    

    sub1_ids = self.lote_primera_sub.select("pair_id").distinct()
    seg2_ids = self.lote_segunda_sub.select("pair_id").distinct()
    
    '''print("pair_id en sub1:", sub1_ids.count())
    print("pair_id en sub2:", seg2_ids.count())
    print(
        "pair_id en sub1 y no en sub2:",
        sub1_ids.join(seg2_ids, on="pair_id", how="left_anti").count()
    )
    print(
        "pair_id en sub2 y no en sub1:",
        seg2_ids.join(sub1_ids, on="pair_id", how="left_anti").count()
    )'''
    
    print("num de las subidas que son únicas en su día y además NO tienen primera validación en el día siguiente:", ids_impredecible_next_day .count())
    print(
        "num de subidas en 1a lote que son unicas y que le hemos sacado 2sub en el next day:",
        self.lote_primera_sub.filter(F.col("n_validaciones_dia") == 1).count()
    )

    return self.df_impredecible_next_day
