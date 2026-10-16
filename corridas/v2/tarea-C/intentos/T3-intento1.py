import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Corpus mínimo del enunciado: 12 documentos, 3 temas reales
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
etiquetas = [c[2] for c in corpus]

stop_words = ["el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una", "por",
              "para", "con", "se", "su", "al", "es", "después", "durante", "contra",
              "donde", "alrededor"]

# Refit del modelo de la Parte 1 (mismos parámetros; determinista con random_state=0)
vec = CountVectorizer(stop_words=stop_words)
X = vec.fit_transform(textos)
lda = LatentDirichletAllocation(n_components=3, learning_method="batch", max_iter=50, random_state=0)
lda.fit(X)

doc_topic = lda.transform(X)  # probabilidades documento-tema (12 x 3)
tema_dom = doc_topic.argmax(axis=1)

# Pureza: por tema LDA, cuenta de la etiqueta real más frecuente entre sus documentos
resumen_temas = {}
acierto = 0
for t in range(3):
    idx = np.where(tema_dom == t)[0]
    cuentas = {}
    for i in idx:
        cuentas[etiquetas[i]] = cuentas.get(etiquetas[i], 0) + 1
    if cuentas:
        etiq_mayor = max(cuentas, key=cuentas.get)
        n_mayor = cuentas[etiq_mayor]
    else:
        etiq_mayor, n_mayor = None, 0
    resumen_temas[str(t)] = {"etiqueta_mayoritaria": etiq_mayor,
                             "cuenta_mayoritaria": int(n_mayor),
                             "total_documentos": int(len(idx)),
                             "cuentas_por_etiqueta": cuentas}
    acierto += n_mayor
pureza = acierto / 12.0

asignacion = []
mal_agrupados = []
for i, did in enumerate(ids):
    t = int(tema_dom[i])
    etiq_mayor = resumen_temas[str(t)]["etiqueta_mayoritaria"]
    bien = etiquetas[i] == etiq_mayor
    asignacion.append({"id": did,
                       "etiqueta_real": etiquetas[i],
                       "tema_lda_dominante": t,
                       "prob_doc_tema": [float(p) for p in doc_topic[i]],
                       "etiqueta_mayoritaria_del_tema": etiq_mayor,
                       "bien_agrupado": bool(bien)})
    if not bien:
        mal_agrupados.append(did)

# Verificación ligera contra la Parte 1 (tamaño del vocabulario)
coincide_t1 = None
try:
    with open("entradas/T1.json", encoding="utf-8") as f:
        t1 = json.load(f)
    coincide_t1 = bool(int(X.shape[1]) == int(t1["tamano_vocabulario"]))
except Exception:
    pass

resultados = {"asignacion_por_documento": asignacion,
              "pureza": float(pureza),
              "documentos_mal_agrupados": mal_agrupados,
              "resumen_temas": resumen_temas,
              "tamano_vocabulario": int(X.shape[1]),
              "coincide_vocab_con_T1": coincide_t1}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("Tabla de asignación (id | etiqueta real | tema LDA dominante | probs doc-tema | bien agrupado):")
for a in asignacion:
    print(f"  {a['id']} | {a['etiqueta_real']:<10} | tema {a['tema_lda_dominante']} | "
          f"{[round(p, 4) for p in a['prob_doc_tema']]} | {a['bien_agrupado']}")
print(f"Pureza: {pureza}")
print(f"Documentos mal agrupados: {mal_agrupados}")
print(f"Tamaño del vocabulario: {X.shape[1]} (coincide con T1: {coincide_t1})")
