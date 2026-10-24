from IPython.display import display, HTML
import pandas as pd
def show_side_by_side(df1, df2, title1="A", title2="B"):
    html = f"""
    <div style="display:flex; gap:20px; align-items:flex-start">
      <div>
        <h4>{title1}</h4>
        {df1.to_html(index=True)}
      </div>
      <div>
        <h4>{title2}</h4>
        {df2.to_html(index=True)}
      </div>
    </div>
    """
    display(HTML(html))

def calcular_coste_por_perfil(x_1sub,x_mal, x_bajadas,unknown,subidas_totales):

    """
    1.sacamos proporciones de aparición de los perfiles
    2.sacamos accuracies y prop de descensos por perfiles
    3.calculamos el la suma de costes, ponderado por la aparicion de perfiles
    """

    
    proporciones = x_1sub["cod_perfil"].value_counts(normalize=True)
    print('proporcion de perfiles x1sub: ',proporciones)

    y_pred_por_perfil = x_bajadas["cod_perfil"].value_counts()
    fallos_por_perfil = x_mal["cod_perfil"].value_counts().reindex(y_pred_por_perfil.index, fill_value=0)

    accs_por_perfil = (y_pred_por_perfil - fallos_por_perfil) / y_pred_por_perfil

    subidas_por_perfil = subidas_totales * proporciones
    unknown_por_perfil = unknown["cod_perfil"].value_counts().reindex(subidas_por_perfil.index, fill_value=0)
    
    prop_descensos_por_perfil = (subidas_por_perfil - unknown_por_perfil) / subidas_por_perfil

    prop_descensos_por_perfil = prop_descensos_por_perfil.reindex(accs_por_perfil.index, fill_value=0)
    proporciones = proporciones.reindex(accs_por_perfil.index, fill_value=0)
    valor_coste = (
    proporciones * (
        prop_descensos_por_perfil * (1 - accs_por_perfil)
        + 0.3 * (1 - prop_descensos_por_perfil)
        )
    ).sum()
    print(f'accuracies por perfil: {accs_por_perfil}')
    print(f'prop desc calculados por perfil: {prop_descensos_por_perfil}')
    
    print(f'valor_coste calculado por como suma de coste por perfiles: {valor_coste}')

    return valor_coste 
        

    