# -*- coding: utf-8 -*-
"""
T2 — Parte 2 — Número de temas (2, 3, 4)

Con el mismo CountVectorizer de la Parte 1 (misma lista de palabras vacías) y los
mismos parámetros de LDA (learning_method='batch', max_iter=50, random_state=0),
se ajusta LatentDirichletAllocation con n_components = 2, 3 y 4 y se reporta la
perplejidad de cada modelo sobre el corpus en una sola tabla comparativa.

Salidas:
  - resultados.json  (perplejidad_k2, perplejidad_k3, perplejidad_k4, tabla_perplejidades, ...)
  - impresión de la tabla comparativa en consola.
  - Sin figura PNG (no se requiere en esta subtarea).
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus: doce oraciones de tres temas (datos del enunciado)
# ----------------------------------------------------------------------
documentos = [
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

# ----------------------------------------------------------------------
# 2. Mismo CountVectorizer de la Parte 1 (misma lista de palabras vacías)
# ----------------------------------------------------------------------
palabras_vacias = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

vectorizador = CountVectorizer(stop_words=palabras_vacias)
X = vectorizador.fit_transform(documentos)
vocabulario = list(vectorizador.get_feature_names_out())
tam_vocab = int(len(vocabulario))
print(f"Tamaño del vocabulario: {tam_vocab}")
print(f"Vocabulario: {vocabulario}")

# ----------------------------------------------------------------------
# 3. Verificación de consistencia con la Parte 1 (entradas/T1.json)
# ----------------------------------------------------------------------
vocab_coincide = None
perp_k3_parte1 = None
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    vocab_t1 = t1.get("vocabulario")
    if vocab_t1 is not None:
        vocab_coincide = bool(list(vocab_t1) == vocabulario)
    if t1.get("perplejidad_k3") is not None:
        perp_k3_parte1 = float(t1["perplejidad_k3"])
    print(f"Vocabulario idéntico al de la Parte 1: {vocab_coincide}")
    if perp_k3_parte1 is not None:
        print(f"Perplejidad k=3 reportada en Parte 1: {perp_k3_parte1}")
except (OSError, ValueError):
    print("Aviso: no se pudo leer entradas/T1.json; se continúa sin verificación cruzada.")

# ----------------------------------------------------------------------
# 4. LDA con k = 2, 3, 4 (mismos parámetros de la Parte 1)
#    La perplejidad se reporta sobre el corpus, tal como pide el enunciado.
# ----------------------------------------------------------------------
perplejidades = {}
for k in (2, 3, 4):
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X)
    perplejidades[k] = float(lda.perplexity(X))
    print(f"k={k}  ->  perplejidad = {perplejidades[k]!r}")

# ----------------------------------------------------------------------
# 5. Tabla comparativa única
# ----------------------------------------------------------------------
tabla = pd.DataFrame(
    {
        "n_componentes": [2, 3, 4],
        "perplejidad": [perplejidades[2], perplejidades[3], perplejidades[4]],
    }
)
mejor_k = int(tabla.loc[tabla["perplejidad"].idxmin(), "n_componentes"])

print("\nTabla comparativa de perplejidades (corpus de 12 oraciones):")
print(tabla.to_string(index=False))
print(f"\nMejor k según perplejidad sobre el corpus: {mejor_k}")
if perp_k3_parte1 is not None:
    diff = float(perplejidades[3]) - perp_k3_parte1
    print(f"Diferencia perplejidad(k=3) actual vs Parte 1: {diff!r}")

# ----------------------------------------------------------------------
# 6. Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "perplejidad_k2": perplejidades[2],
    "perplejidad_k3": perplejidades[3],
    "perplejidad_k4": perplejidades[4],
    "tabla_perplejidades": tabla.to_dict(orient="records"),
    "mejor_k": mejor_k,
    "tamano_vocabulario": tam_vocab,
    "vocabulario_coincide_parte1": vocab_coincide,
}
if perp_k3_parte1 is not None:
    resultados["perplejidad_k3_parte1"] = perp_k3_parte1
    resultados["diferencia_k3_vs_parte1"] = float(perplejidades[3]) - perp_k3_parte1

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nResultados guardados en resultados.json")
