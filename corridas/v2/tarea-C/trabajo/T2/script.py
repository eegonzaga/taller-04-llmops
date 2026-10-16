import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

textos = [
    "El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.",
    "El entrenador del equipo preparó la defensa para el partido final de la liga.",
    "La tenista ganó el torneo después de un partido largo contra la campeona.",
    "El equipo de baloncesto perdió el partido por un punto en el último segundo.",
    "La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.",
    "Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.",
    "El pan se hornea con harina, agua, sal y levadura después de amasar la masa.",
    "La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.",
    "El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.",
    "Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.",
    "La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.",
    "El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.",
]

stop = ["el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una", "por",
        "para", "con", "se", "su", "al", "es", "después", "durante", "contra",
        "donde", "alrededor"]

vec = CountVectorizer(stop_words=stop)
X = vec.fit_transform(textos)

perplejidades = {}
for k in [2, 3, 4]:
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch",
                                    max_iter=50, random_state=0)
    lda.fit(X)
    perplejidades[f"perplejidad_k{k}"] = float(lda.perplexity(X))

tabla = pd.DataFrame({"k": [2, 3, 4],
                      "perplejidad": [perplejidades[f"perplejidad_k{k}"] for k in [2, 3, 4]]})

resultados = {"tamano_vocabulario": int(X.shape[1]), **perplejidades}
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("Tamaño del vocabulario:", X.shape[1])
print(tabla.to_string(index=False))
