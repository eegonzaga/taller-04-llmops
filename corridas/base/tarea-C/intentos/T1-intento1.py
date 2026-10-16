# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — LDA con tres temas
Corpus mínimo de doce oraciones (d01–d12) con tres temas: deporte, cocina, astronomía.
Vectorización con CountVectorizer (lista de palabras vacías del enunciado) y
LatentDirichletAllocation(n_components=3, learning_method='batch', max_iter=50, random_state=0).
Reporta: tamaño del vocabulario, cinco palabras más probables por tema y
perplejidad del modelo sobre el corpus. Guarda todo en resultados.json.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus mínimo (los datos están completos en el enunciado)
# ----------------------------------------------------------------------
corpus = [
    ("d01", "El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.", "deporte"),
    ("d02", "El entrenador del equipo preparó la defensa para el partido final de la liga.", "deporte"),
    ("d03", "La tenista ganó el torneo después de un partido largo contra la campeona.", "deporte"),
    ("d04", "El equipo de baloncesto perdió el partido por un punto en el último segundo.", "deporte"),
    ("d05", "La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.", "cocina"),
    ("d06", "Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.", "cocina"),
    ("d07", "El pan se hornea con harina, agua, sal y levadura después de amasar la masa.", "cocina"),
    ("d08", "La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.", "cocina"),
    ("d09", "El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.", "astronomía"),
    ("d10", "Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.", "astronomía"),
    ("d11", "La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.", "astronomía"),
    ("d12", "El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.", "astronomía"),
]

ids = [c[0] for c in corpus]
textos = [c[1] for c in corpus]
temas_reales = [c[2] for c in corpus]

# ----------------------------------------------------------------------
# 2. Vectorización: CountVectorizer con las palabras vacías del enunciado
#    (resto de parámetros por defecto)
# ----------------------------------------------------------------------
stop_words = ['el', 'la', 'los', 'las', 'de', 'del', 'en', 'y', 'a', 'un', 'una',
              'por', 'para', 'con', 'se', 'su', 'al', 'es', 'después', 'durante',
              'contra', 'donde', 'alrededor']

vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(textos)

vocabulario = vectorizer.get_feature_names_out()
tamano_vocabulario = int(len(vocabulario))

# ----------------------------------------------------------------------
# 3. Ajuste de LDA con k = 3 (hiperparámetros exactos del enunciado)
# ----------------------------------------------------------------------
lda = LatentDirichletAllocation(n_components=3,
                                learning_method='batch',
                                max_iter=50,
                                random_state=0)
lda.fit(X)

# Perplejidad del modelo sobre el corpus, tal como pide el enunciado
perplejidad_k3 = float(lda.perplexity(X))

# ----------------------------------------------------------------------
# 4. Cinco palabras más probables de cada tema
# ----------------------------------------------------------------------
n_top = 5
palabras_top_por_tema = {}
pesos_top_por_tema = {}
for k, componentes in enumerate(lda.components_):
    orden = np.argsort(componentes)[::-1][:n_top]
    palabras_top_por_tema["tema_%d" % (k + 1)] = [str(vocabulario[i]) for i in orden]
    pesos_top_por_tema["tema_%d" % (k + 1)] = [float(componentes[i]) for i in orden]

# Información adicional: mezcla de temas por documento y tema dominante
doc_topic = lda.transform(X)
distribucion_documentos = {ids[i]: [float(v) for v in doc_topic[i]] for i in range(len(ids))}
tema_dominante = {ids[i]: int(np.argmax(doc_topic[i])) + 1 for i in range(len(ids))}

# ----------------------------------------------------------------------
# 5. Guardar todas las cifras en resultados.json (sin redondear)
# ----------------------------------------------------------------------
resultados = {
    "tamano_vocabulario": tamano_vocabulario,
    "palabras_top_por_tema": palabras_top_por_tema,
    "perplejidad_k3": perplejidad_k3,
    "vocabulario": [str(v) for v in vocabulario],
    "pesos_top_por_tema": pesos_top_por_tema,
    "distribucion_documentos_temas": distribucion_documentos,
    "tema_dominante_por_documento": tema_dominante,
    "temas_reales": {ids[i]: temas_reales[i] for i in range(len(ids))},
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6. Reporte por consola
# ----------------------------------------------------------------------
print("=" * 64)
print("Tarea C - Parte 1: LDA con k = 3 sobre el corpus de 12 oraciones")
print("=" * 64)
print("Tamaño del vocabulario: %d" % tamano_vocabulario)
print("Perplejidad (k=3) sobre el corpus: %.10f" % perplejidad_k3)
print("\nCinco palabras más probables por tema:")
for tema in sorted(palabras_top_por_tema.keys()):
    print("  %s: %s" % (tema, ", ".join(palabras_top_por_tema[tema])))
print("\nTema dominante por documento (LDA no ve las etiquetas reales):")
for i in range(len(ids)):
    print("  %s -> tema_%d (tema real: %s)" % (ids[i], tema_dominante[ids[i]], temas_reales[i]))
print("\nResultados guardados en resultados.json")
