# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Parte 1 — Carga de dígitos y partición única
===========================================================
Carga sklearn.datasets.load_digits, reporta número de ejemplos, atributos y
clases, y crea la partición ÚNICA train/test 75%/25% estratificada por clase
con random_state=7. Guarda X_train, X_test, y_train, y_test para reuso en las
siguientes subtareas (dentro de resultados.json y además en particion_digits.npz).
El conjunto de prueba no se usa para ajustar nada.
"""

import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Carga de datos
    # ------------------------------------------------------------------
    digits = load_digits()
    X = np.asarray(digits.data, dtype=float)
    y = np.asarray(digits.target, dtype=int)

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    clases_unicas = np.unique(y)
    n_clases = int(clases_unicas.size)

    # ------------------------------------------------------------------
    # 2) Partición única 75% / 25%, estratificada por clase, random_state=7
    #    (esta es la única partición de toda la tarea; el test no se usa
    #     para ajustar ni elegir nada)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        train_size=0.75,
        stratify=y,
        random_state=7,
    )

    tam_train = int(X_train.shape[0])
    tam_test = int(X_test.shape[0])

    # ------------------------------------------------------------------
    # 3) Verificación de la estratificación (solo descriptivo, no ajusta nada)
    # ------------------------------------------------------------------
    clases, conteo_total = np.unique(y, return_counts=True)
    _, conteo_train = np.unique(y_train, return_counts=True)
    _, conteo_test = np.unique(y_test, return_counts=True)

    distribucion_clases = {}
    for c, ct, ctr, cte in zip(clases, conteo_total, conteo_train, conteo_test):
        distribucion_clases[str(int(c))] = {
            "total": int(ct),
            "train": int(ctr),
            "test": int(cte),
            "prop_train": float(ctr / ct),
            "prop_test": float(cte / ct),
        }

    # ------------------------------------------------------------------
    # 4) Guardar TODAS las cifras en resultados.json (sin redondear)
    #    Se incluyen los arreglos de la partición para reuso posterior.
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1_parte1_carga_y_particion",
        "dataset": "sklearn.datasets.load_digits",
        "n_ejemplos": n_ejemplos,
        "n_atributos": n_atributos,
        "n_clases": n_clases,
        "tam_train": tam_train,
        "tam_test": tam_test,
        "proporcion_train": float(tam_train / n_ejemplos),
        "proporcion_test": float(tam_test / n_ejemplos),
        "random_state": 7,
        "estratificado": True,
        "test_size": 0.25,
        "distribucion_clases": distribucion_clases,
        # Arreglos de la partición única (listas de Python para JSON)
        "X_train": X_train.tolist(),
        "X_test": X_test.tolist(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
    }

    with open("resultados.json", "w") as f:
        json.dump(resultados, f)

    # Copia binaria eficiente de la partición para reuso en subtareas siguientes
    np.savez(
        "particion_digits.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # ------------------------------------------------------------------
    # 5) Impresión de las cifras principales
    # ------------------------------------------------------------------
    print("=" * 60)
    print("T1 - Parte 1: Carga de dígitos y partición única")
    print("=" * 60)
    print(f"n_ejemplos  (total de imágenes)      : {n_ejemplos}")
    print(f"n_atributos (píxeles por imagen 8x8) : {n_atributos}")
    print(f"n_clases    (dígitos 0-9)            : {n_clases}")
    print(f"tam_train   (75%)                    : {tam_train}")
    print(f"tam_test    (25%)                    : {tam_test}")
    print(f"proporción train/test               : "
          f"{tam_train / n_ejemplos:.4f} / {tam_test / n_ejemplos:.4f}")
    print("-" * 60)
    print("Distribución por clase (verificación de estratificación):")
    print(f"{'clase':>5} {'total':>7} {'train':>7} {'test':>7} "
          f"{'%train':>8} {'%test':>8}")
    for c in sorted(distribucion_clases.keys()):
        d = distribucion_clases[c]
        print(f"{c:>5} {d['total']:>7} {d['train']:>7} {d['test']:>7} "
              f"{100 * d['prop_train']:>7.2f}% {100 * d['prop_test']:>7.2f}%")
    print("-" * 60)
    print("Archivos generados:")
    print("  - resultados.json        (cifras + X_train/X_test/y_train/y_test)")
    print("  - particion_digits.npz   (arreglos de la partición única)")
    print("Nota: la partición es la única de toda la tarea; el conjunto de")
    print("prueba no se usa para ajustar ni para elegir nada.")


if __name__ == "__main__":
    main()
