# -*- coding: utf-8 -*-
"""
T3 - Parte 3 - ¿Recupera LDA los temas reales?

Con el modelo de tres temas de la Parte 1 (resultados en entradas/T1.json):
  1) asigna a cada uno de los doce documentos su tema dominante
     (mayor componente de la distribución documento-tema),
  2) calcula la pureza: para cada tema de LDA se cuentan los documentos de
     la etiqueta real más frecuente entre los asignados a él, se suman esas
     cuentas y se dividen por 12,
  3) presenta la asignación por documento (id, etiqueta real, tema LDA) en
     una tabla e identifica los documentos mal agrupados.

Si entradas/T1.json no estuviera disponible o no contuviera la distribución
documento-tema, se reajusta el modelo exacto de la Parte 1 con los
parámetros del enunciado: CountVectorizer con la lista de palabras vacías
dada (resto por defecto) y LatentDirichletAllocation(n_components=3,
learning_method="batch", max_iter=50, random_state=0).

Salidas: resultados.json (asignacion_por_documento, pureza,
documentos_mal_agrupados y cifras de apoyo) e impresión de la tabla.
Esta subtarea NO requiere figura PNG.
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------
# 1) Corpus mínimo del enunciado: 12 oraciones y su etiqueta real.
#    La etiqueta real NO se usa para ajustar LDA (LDA no la ve); solo se
#    emplea para medir la recuperación de los temas (pureza).
# ----------------------------------------------------------------------
CORPUS = [
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

IDS = [c[0] for c in CORPUS]
TEXTOS = [c[1] for c in CORPUS]
ETIQUETAS_REALES = [c[2] for c in CORPUS]
N_DOCS = len(CORPUS)   # 12
N_TEMAS = 3

STOP_WORDS = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

RUTA_T1 = os.path.join("entradas", "T1.json")


# ----------------------------------------------------------------------
# 2) Lectura de la Parte 1 (entradas/T1.json), tolerante a formatos
# ----------------------------------------------------------------------
def cargar_t1():
    """Devuelve el contenido de entradas/T1.json o None si no se puede leer."""
    if not os.path.isfile(RUTA_T1):
        return None
    try:
        with open(RUTA_T1, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _secuencia_por_documento(valor):
    """Ordena un valor de T1.json en el orden d01..d12 si es posible."""
    if isinstance(valor, dict):
        if all(i in valor for i in IDS):
            return [valor[i] for i in IDS]
        if all(str(k) in valor for k in range(N_DOCS)):
            return [valor[str(k)] for k in range(N_DOCS)]
        return None
    if isinstance(valor, (list, tuple)):
        return list(valor)
    return None


def extraer_distribucion(t1):
    """Matriz documento-tema (12 x 3) desde T1.json (filas normalizadas)."""
    if not isinstance(t1, dict):
        return None
    filas = _secuencia_por_documento(t1.get("distribucion_documentos_temas"))
    if filas is None:
        return None
    try:
        arr = np.asarray(filas, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (N_DOCS, N_TEMAS):
        return None
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        return None
    totales = arr.sum(axis=1, keepdims=True)
    totales[totales == 0.0] = 1.0
    return arr / totales  # normalizar filas (el argmax no cambia)


def extraer_dominante_t1(t1):
    """Tema dominante por documento guardado en T1.json (verificación)."""
    if not isinstance(t1, dict):
        return None
    seq = _secuencia_por_documento(t1.get("tema_dominante_por_documento"))
    if seq is None:
        return None
    try:
        arr = np.asarray([int(v) for v in seq], dtype=int)
    except (TypeError, ValueError):
        return None
    if arr.shape != (N_DOCS,):
        return None
    return arr


def extraer_etiquetas_t1(t1):
    """Etiquetas reales guardadas en T1.json (verificación), o None."""
    if not isinstance(t1, dict):
        return None
    seq = _secuencia_por_documento(t1.get("temas_reales"))
    if seq is None or len(seq) != N_DOCS:
        return None
    return [str(x) for x in seq]


def reajustar_lda_parte1():
    """Reajusta el modelo exacto de la Parte 1 (respaldo si falta T1.json)."""
    vectorizer = CountVectorizer(stop_words=STOP_WORDS)  # resto por defecto
    X = vectorizer.fit_transform(TEXTOS)
    lda = LatentDirichletAllocation(
        n_components=N_TEMAS,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X)
    return lda.transform(X)


def main():
    # ------------------------------------------------------------------
    # 3) Distribución documento-tema del modelo de tres temas de la Parte 1
    # ------------------------------------------------------------------
    t1 = cargar_t1()
    dist_t1 = extraer_distribucion(t1)
    dominante_t1 = extraer_dominante_t1(t1)
    etiquetas_t1 = extraer_etiquetas_t1(t1)

    if dist_t1 is not None:
        dist_doc_topic = dist_t1
        fuente = "entradas/T1.json :: distribucion_documentos_temas (modelo de la Parte 1)"
    else:
        dist_doc_topic = reajustar_lda_parte1()
        fuente = ("respaldo: reajuste de la Parte 1 con CountVectorizer(stop_words) y "
                  "LatentDirichletAllocation(n_components=3, learning_method='batch', "
                  "max_iter=50, random_state=0)")

    # Tema dominante = mayor componente de la distribución documento-tema
    tema_dominante = np.argmax(dist_doc_topic, axis=1).astype(int)

    # Verificaciones de coherencia con la Parte 1 (informativas)
    coincide_dominante_t1 = None
    if dominante_t1 is not None:
        coincide_dominante_t1 = bool(np.array_equal(tema_dominante, dominante_t1))

    coincide_reajuste = None
    if dist_t1 is not None:
        try:
            coincide_reajuste = bool(np.array_equal(
                tema_dominante, np.argmax(reajustar_lda_parte1(), axis=1)))
        except Exception:
            coincide_reajuste = None

    coinciden_etiquetas_t1 = None
    if etiquetas_t1 is not None:
        if all(x in set(ETIQUETAS_REALES) for x in etiquetas_t1):
            coinciden_etiquetas_t1 = bool(etiquetas_t1 == ETIQUETAS_REALES)

    # ------------------------------------------------------------------
    # 4) Pureza: por cada tema LDA, cuenta de la etiqueta real mayoritaria
    #    entre los documentos asignados a él; suma / 12.
    # ------------------------------------------------------------------
    resumen_temas = {}
    for k in range(N_TEMAS):
        asignados = [i for i in range(N_DOCS) if int(tema_dominante[i]) == k]
        cuentas = Counter(ETIQUETAS_REALES[i] for i in asignados)
        if cuentas:
            # mayoría determinista: mayor cuenta; empate -> orden alfabético
            etiqueta_may, cuenta_may = sorted(
                cuentas.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        else:
            etiqueta_may, cuenta_may = None, 0
        resumen_temas[k] = {
            "documentos_asignados": [IDS[i] for i in asignados],
            "cuentas_por_etiqueta_real": {e: int(c) for e, c in sorted(cuentas.items())},
            "etiqueta_real_mayoritaria": etiqueta_may,
            "cuenta_mayoritaria": int(cuenta_may),
        }

    aciertos = int(sum(resumen_temas[k]["cuenta_mayoritaria"] for k in range(N_TEMAS)))
    pureza = aciertos / float(N_DOCS)

    # ------------------------------------------------------------------
    # 5) Asignación por documento y documentos mal agrupados
    #    (mal agrupado: su etiqueta real difiere de la mayoría del tema
    #     LDA al que fue asignado)
    # ------------------------------------------------------------------
    asignacion_por_documento = []
    documentos_mal_agrupados = []
    for i in range(N_DOCS):
        k = int(tema_dominante[i])
        etiqueta_may = resumen_temas[k]["etiqueta_real_mayoritaria"]
        bien = (etiqueta_may is not None) and (ETIQUETAS_REALES[i] == etiqueta_may)
        asignacion_por_documento.append({
            "id": IDS[i],
            "etiqueta_real": ETIQUETAS_REALES[i],
            "tema_lda": k,
            "p_tema_dominante": float(np.max(dist_doc_topic[i])),
            "distribucion_documento_temas": [float(p) for p in dist_doc_topic[i]],
            "etiqueta_mayoritaria_del_tema": etiqueta_may,
            "bien_agrupado": bool(bien),
        })
        if not bien:
            documentos_mal_agrupados.append({
                "id": IDS[i],
                "etiqueta_real": ETIQUETAS_REALES[i],
                "tema_lda": k,
                "etiqueta_mayoritaria_del_tema": etiqueta_may,
                "p_tema_dominante": float(np.max(dist_doc_topic[i])),
                "distribucion_documento_temas": [float(p) for p in dist_doc_topic[i]],
            })

    # ------------------------------------------------------------------
    # 6) Guardado de todas las cifras en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "asignacion_por_documento": asignacion_por_documento,
        "pureza": float(pureza),
        "documentos_mal_agrupados": documentos_mal_agrupados,
        "ids_mal_agrupados": [m["id"] for m in documentos_mal_agrupados],
        "tema_dominante_por_documento": [int(t) for t in tema_dominante],
        "distribucion_documento_tema_usada": [
            [float(p) for p in fila] for fila in dist_doc_topic
        ],
        "aciertos_de_mayoria": aciertos,
        "n_documentos": int(N_DOCS),
        "n_temas": int(N_TEMAS),
        "resumen_por_tema_lda": {str(k): resumen_temas[k] for k in range(N_TEMAS)},
        "fuente_distribucion_documento_tema": fuente,
        "coincide_con_tema_dominante_de_T1": coincide_dominante_t1,
        "coincide_con_reajuste_local_de_la_Parte1": coincide_reajuste,
        "coinciden_etiquetas_reales_de_T1": coinciden_etiquetas_t1,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 7) Salida por pantalla
    # ------------------------------------------------------------------
    print("=" * 78)
    print("T3 - Parte 3 - ¿Recupera LDA los temas reales?")
    print("=" * 78)
    print(f"Fuente de la distribución documento-tema: {fuente}")
    if coincide_dominante_t1 is False:
        print("Aviso: el argmax de la distribución difiere de 'tema_dominante_por_documento'")
        print("de T1.json; se usa el argmax (definición de la Parte 3).")

    tabla = pd.DataFrame([{
        "id": a["id"],
        "etiqueta_real": a["etiqueta_real"],
        "tema_lda": a["tema_lda"],
        "p_tema_dominante": round(a["p_tema_dominante"], 4),
        "mayoria_del_tema": a["etiqueta_mayoritaria_del_tema"],
        "bien_agrupado": "sí" if a["bien_agrupado"] else "NO",
    } for a in asignacion_por_documento])
    print("\nAsignación por documento (tema dominante = mayor componente de theta_d):")
    print(tabla.to_string(index=False))

    print("\nComposición real de cada tema LDA:")
    for k in range(N_TEMAS):
        r = resumen_temas[k]
        docs = ", ".join(r["documentos_asignados"]) if r["documentos_asignados"] else "-"
        print(f"  Tema LDA {k}: cuentas={r['cuentas_por_etiqueta_real']} | "
              f"mayoría='{r['etiqueta_real_mayoritaria']}' "
              f"({r['cuenta_mayoritaria']}/{len(r['documentos_asignados'])}) | docs: {docs}")

    print(f"\nPureza = {aciertos}/{N_DOCS} = {pureza:.6f} ({100.0 * pureza:.2f}%)")

    if documentos_mal_agrupados:
        print(f"\nDocumentos mal agrupados ({len(documentos_mal_agrupados)}):")
        for m in documentos_mal_agrupados:
            dists = ", ".join(f"{p:.4f}" for p in m["distribucion_documento_temas"])
            print(f"  - {m['id']} | etiqueta real: {m['etiqueta_real']} | "
                  f"tema LDA: {m['tema_lda']} | mayoría del tema: "
                  f"'{m['etiqueta_mayoritaria_del_tema']}' | p(temas)=[{dists}]")
        print("  Nota: LDA no observa las etiquetas reales y los índices de sus temas son")
        print("  arbitrarios; los fallos provienen del vocabulario compartido entre temas.")
    else:
        print("\nDocumentos mal agrupados: ninguno; LDA recupera exactamente los tres temas reales.")

    print("\nCifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
