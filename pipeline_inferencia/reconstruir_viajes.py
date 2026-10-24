import pandas as pd

def reconstruir_viajes(x_checked: pd.DataFrame):

    df = x_checked.copy().reset_index(drop=True)
    df["timestamp_validacion"] = pd.to_datetime(df["timestamp_validacion"], errors="coerce", utc=True)
    df = df.sort_values(["id_tarjeta", "timestamp_validacion"]).reset_index(drop=True)

    df_unlinked = df[df["tipo"].eq("UNLINKED")].copy()

    df = df[~df["tipo"].eq("UNLINKED")].copy().reset_index(drop=True)

    df["fecha"] = df["timestamp_validacion"].dt.date
    df["fecha_str"] = df["timestamp_validacion"].dt.strftime("%d/%m/%Y")

    X_list, Y_list = [], []
    prev_tarjeta, prev_fecha, prev_tipo = None, None, None
    X = 0
    Y = 0

    for _, row in df.iterrows():
        tarjeta = row["id_tarjeta"]
        fecha = row["fecha"]
        tipo = row["tipo"]

        new_day_or_card = (tarjeta != prev_tarjeta) or (fecha != prev_fecha)

        if new_day_or_card:
            X = 1
            Y = 1
        else:
            if prev_tipo == "BAJADA":
                X += 1
                Y = 1
            else:
                Y += 1

        X_list.append(X)
        Y_list.append(Y)

        prev_tarjeta, prev_fecha, prev_tipo = tarjeta, fecha, tipo

    df["X_viaje_dia"] = X_list
    df["Y_etapa"] = Y_list

    df["id_parada"] = (
        df["id_tarjeta"].astype(str) + ":" +
        df["fecha_str"] + ":" +
        df["X_viaje_dia"].astype(str) + "_" + df["Y_etapa"].astype(str)
    )

    df_viajes = (
        df.groupby(["id_tarjeta", "fecha_str", "X_viaje_dia"], as_index=False)
          .agg(
              fecha_hora_inicio=("timestamp_validacion", "min"),
              fecha_hora_fin=("timestamp_validacion", "max"),
          )
    )
    df_viajes["tipo"] = "RECONSTRUIDO"
    df_viajes["tarjeta"] = df_viajes["id_tarjeta"].astype(str)
    df_viajes["id"] = (
        df_viajes["id_tarjeta"].astype(str) + ":" +
        df_viajes["fecha_str"] + ":" +
        df_viajes["X_viaje_dia"].astype(str)
    )

    df_viajes = df_viajes[["id", "fecha_hora_inicio", "fecha_hora_fin", "tipo", "tarjeta"]].copy()

    return df_unlinked, df, df_viajes

def reconstruir_od(x_checked: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con una fila por viaje completo:
      id_tarjeta | fecha_str | X_viaje_dia |
      origen_punto_acceso | destino_punto_acceso | timestamp_inicio

    El origen es la parada de subida del primer tramo del viaje y el destino
    es la prediccion_descenso del último tramo (siempre una BAJADA).
    Los UNLINKED (prediccion_descenso == 'unknown') quedan excluidos por
    construcción: reconstruir_viajes() los separa antes de asignar etapas.
    """
    df_unlinked, df, _ = reconstruir_viajes(x_checked)

    key = ["id_tarjeta", "fecha_str", "X_viaje_dia"]
    df_sorted = df.sort_values(key + ["Y_etapa"])

    primero = (
        df_sorted.groupby(key, sort=False).first().reset_index()
        [key + ["punto_acceso", "timestamp_validacion"]]
        .rename(columns={
            "punto_acceso": "origen_punto_acceso",
            "timestamp_validacion": "timestamp_inicio",
        })
    )
    ultimo = (
        df_sorted.groupby(key, sort=False).last().reset_index()
        [key + ["prediccion_descenso"]]
        .rename(columns={"prediccion_descenso": "destino_punto_acceso"})
    )

    return primero.merge(ultimo, on=key), df_unlinked
