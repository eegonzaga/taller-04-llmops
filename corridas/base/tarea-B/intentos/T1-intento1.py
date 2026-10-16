# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Parte 1 — Atención escalada en NumPy
Implementa softmax numéricamente estable (restando el máximo de cada fila) y
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V con d_k = 4,
usando las matrices del enunciado (Q: 2x4, K: 3x4, V: 3x2).
Guarda todas las cifras en resultados.json (precisión completa) y las imprime
con cuatro decimales. No se requiere figura PNG.
"""

import json
import numpy as np

# ----------------------------------------------------------------------
# 1) Datos del enunciado
# ----------------------------------------------------------------------
Q = np.array([[1.0, 0.0, 1.0, 0.5],
              [0.0, 2.0, 0.0, 1.0]], dtype=float)          # (2, 4)

K = np.array([[1.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 1.0, 0.5],
              [1.0, 0.0, 1.0, 1.0]], dtype=float)          # (3, 4)

V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [2.0, 2.0]], dtype=float)                    # (3, 2)

d_k = Q.shape[1]  # d_k = 4

# ----------------------------------------------------------------------
# 2) Softmax numéricamente estable (por filas, restando el máximo)
# ----------------------------------------------------------------------
def softmax_estable(Z):
    """Softmax por filas, estable: exp(z - max_fila) / suma, evita overflow."""
    Z = np.asarray(Z, dtype=float)
    Z_max = np.max(Z, axis=1, keepdims=True)      # máximo de cada fila
    E = np.exp(Z - Z_max)                          # exponenciales desplazadas
    return E / np.sum(E, axis=1, keepdims=True)    # normalización por fila

# ----------------------------------------------------------------------
# 3) Atención escalada: softmax(Q K^T / sqrt(d_k)) V
# ----------------------------------------------------------------------
puntajes = Q @ K.T                       # (2, 3)  productos punto Q·K_j
escalados = puntajes / np.sqrt(d_k)      # (2, 3)  división por sqrt(4) = 2
pesos = softmax_estable(escalados)       # (2, 3)  matriz de pesos de atención
salida = pesos @ V                       # (2, 2)  salida de la atención

# Verificación de formas
assert puntajes.shape == (2, 3)
assert pesos.shape == (2, 3)
assert salida.shape == (2, 2)

# ----------------------------------------------------------------------
# 4) Guardar TODAS las cifras en resultados.json (precisión completa)
# ----------------------------------------------------------------------
resultados = {
    "d_k": int(d_k),
    "sqrt_d_k": float(np.sqrt(d_k)),
    "Q_2x4": Q.tolist(),
    "K_3x4": K.tolist(),
    "V_3x2": V.tolist(),
    "puntajes_QKt_2x3": puntajes.tolist(),
    "puntajes_escalados_2x3": escalados.tolist(),
    "pesos_atencion_2x3": pesos.tolist(),
    "salida_atencion_2x2": salida.tolist(),
    "suma_filas_pesos": pesos.sum(axis=1).tolist(),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 5) Impresión de las cifras principales con cuatro decimales
# ----------------------------------------------------------------------
def fmt_mat(M):
    """Formatea una matriz con cuatro decimales por elemento."""
    return "\n".join("  [" + "  ".join(f"{v:.4f}" for v in fila) + "]"
                     for fila in np.asarray(M))

print("=" * 60)
print("Atención escalada: softmax(Q K^T / sqrt(d_k)) V, con d_k =", d_k)
print("=" * 60)
print("Q (2x4):")
print(fmt_mat(Q))
print("K (3x4):")
print(fmt_mat(K))
print("V (3x2):")
print(fmt_mat(V))
print(f"sqrt(d_k) = {np.sqrt(d_k):.4f}")
print("-" * 60)
print("Puntajes Q K^T (2x3):")
print(fmt_mat(puntajes))
print("Puntajes escalados Q K^T / sqrt(d_k) (2x3):")
print(fmt_mat(escalados))
print("-" * 60)
print("Pesos de atención softmax(Q K^T / sqrt(d_k)) (2x3):")
print(fmt_mat(pesos))
print("Suma de cada fila de pesos (debe ser 1.0000):")
print("  [" + "  ".join(f"{s:.4f}" for s in pesos.sum(axis=1)) + "]")
print("-" * 60)
print("Salida Attention(Q, K, V) (2x2):")
print(fmt_mat(salida))
print("=" * 60)
print("Resultados guardados en resultados.json (precisión completa).")
