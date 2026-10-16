# -*- coding: utf-8 -*-
"""
T2 - Parte 2 — GaussianNB vs Regresión Logística sobre el conjunto de prueba de T1.

- GaussianNB (valores por defecto) entrenado sobre los atributos ORIGINALES, sin escalar.
- LogisticRegression(max_iter=3000) entrenado sobre atributos ESTANDARIZADOS con un
  StandardScaler ajustado SOLO con el conjunto de entrenamiento.
- Ambos modelos se evalúan sobre el conjunto de prueba definido en T1 (entradas/T1.json).
- Cifras guardadas en resultados.json e impresas en una tabla comparativa.
- No se requiere figura PNG para esta subtarea.
"""

import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def main():
    # ------------------------------------------------------------------
    # 1) Cargar la partición definida en T1
    # ------------------------------------------------------------------
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)

    random_state = int(t1.get("random_state", 7))
    test_size = float(t1.get("test_size", 0.25))

    # ------------------------------------------------------------------
    # 2) Cargar el conjunto de dígitos manuscritos de scikit-learn
    # ------------------------------------------------------------------
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train_raw = t1.get("X_train", None)
    X_test_raw = t1.get("X_test", None)

    if X_train_raw is None or X_test_raw is None:
        # Respaldo: reproducir la partición de T1 con sus mismos parámetros
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train_raw = np.asarray(X_train_raw)
        X_test_raw = np.asarray(X_test_raw)
        es_indices = (X_train_raw.ndim == 1) or (
            X_train_raw.ndim == 2 and X_train_raw.shape[1] == 1
        )
        if es_indices:
            # T1 guardó los ÍNDICES de entrenamiento y prueba
            train_idx = X_train_raw.ravel().astype(int)
            test_idx = X_test_raw.ravel().astype(int)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        else:
            # T1 guardó las MATRICES de atributos; recuperar etiquetas por coincidencia
            lookup = {X[i].tobytes(): int(y[i]) for i in range(X.shape[0])}
            try:
                X_train = X_train_raw.astype(float)
                X_test = X_test_raw.astype(float)
                y_train = np.array(
                    [lookup[fila.tobytes()] for fila in X_train], dtype=int
                )
                y_test = np.array(
                    [lookup[fila.tobytes()] for fila in X_test], dtype=int
                )
            except KeyError:
                # Respaldo: reproducir la partición de T1 con sus mismos parámetros
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

    # ------------------------------------------------------------------
    # 3) Modelo generativo: GaussianNB sobre atributos originales (sin escalar)
    # ------------------------------------------------------------------
    gnb = GaussianNB()  # valores por defecto
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    accuracy_gnb = float(accuracy_score(y_test, y_pred_gnb))
    f1_macro_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))

    # ------------------------------------------------------------------
    # 4) Modelo discriminativo: LogisticRegression sobre atributos estandarizados
    #    (StandardScaler ajustado SOLO con el conjunto de entrenamiento)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)   # ajuste solo con entrenamiento
    X_test_std = scaler.transform(X_test)         # la prueba solo se transforma

    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train_std, y_train)
    y_pred_lr = lr.predict(X_test_std)
    accuracy_lr = float(accuracy_score(y_test, y_pred_lr))
    f1_macro_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

    # ------------------------------------------------------------------
    # 5) Guardar TODAS las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2 - Parte 2 - GaussianNB vs regresion logistica en prueba",
        "dataset": "sklearn.datasets.load_digits",
        "tam_train": int(X_train.shape[0]),
        "tam_test": int(X_test.shape[0]),
        "n_atributos": int(X_train.shape[1]),
        "n_clases": int(len(np.unique(y))),
        "random_state": int(random_state),
        "accuracy_gnb": accuracy_gnb,
        "f1_macro_gnb": f1_macro_gnb,
        "accuracy_lr": accuracy_lr,
        "f1_macro_lr": f1_macro_lr,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6) Imprimir las cifras principales y la tabla comparativa
    # ------------------------------------------------------------------
    print("Cifras principales:")
    print("  accuracy_gnb =", accuracy_gnb)
    print("  f1_macro_gnb =", f1_macro_gnb)
    print("  accuracy_lr  =", accuracy_lr)
    print("  f1_macro_lr  =", f1_macro_lr)
    print()

    print("Tabla: GaussianNB vs Regresión Logística (conjunto de prueba)")
    print("-" * 62)
    print(f"{'Modelo':<42}{'Accuracy':>10}{'F1 macro':>10}")
    print("-" * 62)
    print(
        f"{'GaussianNB (atributos sin escalar)':<42}"
        f"{accuracy_gnb:>10.4f}{f1_macro_gnb:>10.4f}"
    )
    print(
        f"{'LogisticRegression (atributos escalados)':<42}"
        f"{accuracy_lr:>10.4f}{f1_macro_lr:>10.4f}"
    )
    print("-" * 62)
    print("Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
