# -*- coding: utf-8 -*-
"""
T3 - Parte 3 — Curva de aprendizaje de ambos modelos.

Con la partición única de T1 (75% train / 25% test, estratificada,
random_state=7) y los dos modelos de T2:
  * GaussianNB() (valores por defecto) sobre los atributos originales, sin escalar.
  * LogisticRegression(max_iter=3000) sobre atributos estandarizados con un
    StandardScaler ajustado SOLO con los datos de entrenamiento de cada corrida.

Se entrena con m = 20, 50, 100, 200 y 400 ejemplos (submuestras estratificadas
con train_test_split(X_train, y_train, train_size=m, stratify=y_train,
random_state=7)) y con el entrenamiento completo. La evaluación se hace SIEMPRE
sobre el mismo conjunto de prueba de T1 (nunca se usa para ajustar nada).

Salidas:
  * resultados.json  -> accuracy_gnb_por_m, accuracy_lr_por_m, archivo_figura_png, ...
  * curva_aprendizaje_T3_parte3.png -> exactitud vs número de ejemplos (una curva por modelo)
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ARCHIVO_RESULTADOS = "resultados.json"
ARCHIVO_FIGURA = "curva_aprendizaje_T3_parte3.png"
VALORES_M = [20, 50, 100, 200, 400]
RANDOM_STATE_SUBMUESTRA = 7

# ----------------------------------------------------------------------
# 1) Recuperar la partición única de T1
#    (se regenera con los parámetros de T1 y se verifica contra entradas/T1.json)
# ----------------------------------------------------------------------
digits = load_digits()
X, y = digits.data, digits.target

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)

X_train, X_test, y_train, y_test = X_tr, X_te, y_tr, y_te
origen_particion = (
    "regenerada con los parámetros de T1 (test_size=0.25, estratificada, random_state=7)"
)

try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    if "X_train" in t1 and "X_test" in t1:
        X_train_t1 = np.asarray(t1["X_train"], dtype=float)
        X_test_t1 = np.asarray(t1["X_test"], dtype=float)
        if (
            X_train_t1.shape == X_tr.shape
            and X_test_t1.shape == X_te.shape
            and np.allclose(X_train_t1, X_tr)
            and np.allclose(X_test_t1, X_te)
        ):
            # La partición de T1 coincide exactamente con la regenerada:
            # usamos los arreglos de T1 con las etiquetas correspondientes.
            X_train, X_test = X_train_t1, X_test_t1
            origen_particion = "entradas/T1.json (verificada idéntica a la partición canónica)"
except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
    pass

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

# ----------------------------------------------------------------------
# 2) Entrenar y evaluar cada modelo en cada tamaño de entrenamiento
# ----------------------------------------------------------------------

def entrenar_y_evaluar(Xs, ys):
    """Entrena los dos modelos de T2 sobre (Xs, ys) y evalúa sobre el
    conjunto de prueba fijo. Devuelve (accuracy_gnb, accuracy_lr)."""
    # Generativo: GaussianNB sobre los atributos originales, sin escalar
    gnb = GaussianNB()
    gnb.fit(Xs, ys)
    acc_g = float(accuracy_score(y_test, gnb.predict(X_test)))

    # Discriminativo: LR sobre atributos estandarizados; el escalador se
    # ajusta SOLO con los datos de entrenamiento de esta corrida
    scaler = StandardScaler().fit(Xs)
    lr = LogisticRegression(max_iter=3000)
    lr.fit(scaler.transform(Xs), ys)
    acc_l = float(accuracy_score(y_test, lr.predict(scaler.transform(X_test))))

    return acc_g, acc_l


accuracy_gnb_por_m = {}
accuracy_lr_por_m = {}
tamanos_efectivos = []

for m in VALORES_M:
    if m >= n_train:  # salvaguarda (no ocurre con los m pedidos)
        X_sub, y_sub = X_train, y_train
    else:
        X_sub, _, y_sub, _ = train_test_split(
            X_train,
            y_train,
            train_size=m,
            stratify=y_train,
            random_state=RANDOM_STATE_SUBMUESTRA,
        )
    tamanos_efectivos.append(int(X_sub.shape[0]))
    acc_g, acc_l = entrenar_y_evaluar(X_sub, y_sub)
    accuracy_gnb_por_m[str(m)] = acc_g
    accuracy_lr_por_m[str(m)] = acc_l

# Entrenamiento completo (mismo conjunto de entrenamiento de T1/T2)
acc_gnb_completo, acc_lr_completo = entrenar_y_evaluar(X_train, y_train)
accuracy_gnb_por_m["completo"] = acc_gnb_completo
accuracy_lr_por_m["completo"] = acc_lr_completo

# ----------------------------------------------------------------------
# 3) Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T3 - Parte 3 - Curva de aprendizaje",
    "dataset": "sklearn.datasets.load_digits",
    "particion_usada": origen_particion,
    "tam_train": n_train,
    "tam_test": n_test,
    "valores_m": list(VALORES_M),
    "tamanos_efectivos_submuestra": tamanos_efectivos,
    "random_state_submuestra": RANDOM_STATE_SUBMUESTRA,
    "accuracy_gnb_por_m": accuracy_gnb_por_m,
    "accuracy_lr_por_m": accuracy_lr_por_m,
    "accuracy_gnb_completo": acc_gnb_completo,
    "accuracy_lr_completo": acc_lr_completo,
    "archivo_figura_png": ARCHIVO_FIGURA,
}

with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 4) Figura: exactitud contra número de ejemplos (una curva por modelo)
# ----------------------------------------------------------------------
ms_plot = list(VALORES_M) + [n_train]
gnb_plot = [accuracy_gnb_por_m[str(m)] for m in VALORES_M] + [acc_gnb_completo]
lr_plot = [accuracy_lr_por_m[str(m)] for m in VALORES_M] + [acc_lr_completo]

plt.figure(figsize=(8, 5))
plt.plot(ms_plot, gnb_plot, "o-", color="tab:blue",
         label="Naive Bayes gaussiano (generativo)")
plt.plot(ms_plot, lr_plot, "s-", color="tab:orange",
         label="Regresión logística (discriminativo)")
plt.xscale("log")
plt.xticks(ms_plot, [str(v) for v in ms_plot])
plt.xlabel("Número de ejemplos de entrenamiento (m, escala logarítmica)")
plt.ylabel("Exactitud (accuracy) en el conjunto de prueba")
plt.title("Curva de aprendizaje: generativo vs. discriminativo (dígitos, random_state=7)")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ARCHIVO_FIGURA, dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 5) Imprimir las cifras principales
# ----------------------------------------------------------------------
print("=" * 64)
print("T3 - Parte 3: Curva de aprendizaje (GNB vs. Regresión Logística)")
print("=" * 64)
print(f"Partición usada: {origen_particion}")
print(f"tam_train={n_train} | tam_test={n_test} | evaluación fija en prueba")
print(f"Submuestras: train_size=m, stratify=y_train, random_state={RANDOM_STATE_SUBMUESTRA}")
print(f"Tamaños efectivos de submuestra: {tamanos_efectivos}")
print("-" * 64)
print(f"{'m':>10} | {'acc GNB':>10} | {'acc LR':>10}")
for m in VALORES_M:
    print(f"{m:>10} | {accuracy_gnb_por_m[str(m)]:>10.4f} | {accuracy_lr_por_m[str(m)]:>10.4f}")
print(f"{'completo':>10} | {acc_gnb_completo:>10.4f} | {acc_lr_completo:>10.4f}")
print("-" * 64)
print(f"Figura guardada en: {ARCHIVO_FIGURA}")
print(f"Resultados guardados en: {ARCHIVO_RESULTADOS}")
