import json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Corpus: doce oraciones con etiqueta real (LDA no la usa)
datos = [
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
ids = [d[0] for d in datos]
textos = [d[1] for d in datos]
etiquetas = [d[2] for d in datos]

stop_words = ("el la los las de del en y a un una por para con se su al es "
              "después durante contra donde alrededor").split()

# Vectorización (parámetros por defecto salvo stop_words)
vec = CountVectorizer(stop_words=stop_words)
X = vec.fit_transform(textos)

# LDA con tres temas
lda = LatentDirichletAllocation(n_components=3, learning_method="batch",
                                max_iter=50, random_state=0)
lda.fit(X)

vocab = vec.get_feature_names_out().tolist()
tamano_vocabulario = int(len(vocab))

palabras_top_por_tema = []
for comp in lda.components_:
    idx = np.argsort(comp)[::-1][:5]
    palabras_top_por_tema.append([vocab[i] for i in idx])

perplejidad_lda3 = float(lda.perplexity(X))

resultados = {
    "tamano_vocabulario": tamano_vocabulario,
    "palabras_top_por_tema": palabras_top_por_tema,
    "perplejidad_lda3": perplejidad_lda3,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# Artefactos para las Partes 2 y 3 (sin pickle: JSON + npz)
sparse.save_npz("matriz_dt.npz", X)
with open("vectorizador.json", "w", encoding="utf-8") as f:
    json.dump({"vocabulario": vocab, "ids": ids, "etiquetas": etiquetas,
               "stop_words": stop_words}, f, ensure_ascii=False, indent=2)
with open("modelo_lda3.json", "w", encoding="utf-8") as f:
    json.dump({"componentes": lda.components_.tolist(),
               "doc_topic_prior": float(lda.doc_topic_prior_),
               "topic_word_prior": float(lda.topic_word_prior_)},
              f, ensure_ascii=False, indent=2)

print("Tamaño del vocabulario:", tamano_vocabulario)
for t, palabras in enumerate(palabras_top_por_tema):
    print(f"Tema {t}: {', '.join(palabras)}")
print("Perplejidad LDA (3 temas):", perplejidad_lda3)
