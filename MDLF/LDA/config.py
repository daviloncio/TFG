# config.py — hyperparameters for RegionFunctionLDA
#
# n_topics      : K, dimensión del vector de función regional.
#                 Valores a probar: 8, 16, 32.
#
# slot_minutes  : tamaño del bin temporal en minutos.
#                 15 → 96 bins/día  |  30 → 48 bins/día
#                 Si los tópicos quedan demasiado ruidosos, subir a 30.
#
# alpha         : prior doc-topic (sklearn: doc_topic_prior).
#                 Valor bajo → cada estación se especializa en pocos tópicos.
#                 None = sklearn usa 1/K por defecto.
#                 Rango a explorar: [None, 0.1, 0.5, 1.0]
#
# beta          : prior topic-word (sklearn: topic_word_prior).
#                 Valor bajo → cada tópico es más "puro" (pocas palabras dominan).
#                 None = sklearn usa 1/K por defecto.
#                 Rango a explorar: [None, 0.01, 0.1]
#
# max_iter      : iteraciones del algoritmo variacional EM.
#                 20 suele bastar; subir a 50 si la convergencia es lenta.
#
# n_top_tokens  : cuántos tokens mostrar por tópico en la inspección visual.

DEFAULTS = {
    "n_topics": 16,
    "slot_minutes": 15,
    "alpha": None,
    "beta": None,
    "max_iter": 20,
    "random_state": 42,
    "n_top_tokens": 15,
}
