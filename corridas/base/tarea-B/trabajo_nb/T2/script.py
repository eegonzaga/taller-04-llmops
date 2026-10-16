# -*- coding: utf-8 -*-
"""
SUBTAREA T2 — Parte 2 — Efecto de escalar por sqrt(d_k)
--------------------------------------------------------
Crea una sola vez rng = np.random.default_rng(0). Para d_k = 4, 64 y 512 (en ese
orden) toma q = rng.standard_normal(d_k) y después K = rng.standard_normal((10, d_k)),
calcula los pesos softmax de los puntajes K @ q sin escalar y escalados por sqrt(d_k),
y reporta el peso máximo y la entropía en bits de cada distribución.

Guarda todas las cifras en resultados.json y las imprime en una tabla con 4 decimales.
No se requiere figura PNG para esta subtarea.
"""

import json
import numpy as np


def softmax_estable(x):
    """Softmax numéricamente estable: resta el máximo antes de exponenciar."""
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -sum(p * log2(p)), con 0*log2(0) = 0."""
    p = np.asarray(p, dtype=float)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


# ------------------------------------------------------------------
# Generador aleatorio: se crea UNA sola vez, como pide el enunciado
# ------------------------------------------------------------------
rng = np.random.default_rng(0)

d_k_lista = [4, 64, 512]  # orden exigido por el enunciado

peso_maximo_sin_escalar_por_dk = {}
entropia_bits_sin_escalar_por_dk = {}
peso_maximo_escalado_por_dk = {}
entropia_bits_escalado_por_dk = {}
tabla_filas = []

for d_k in d_k_lista:
    # Orden estricto de los sorteos: primero q, después K (mismo rng, sin reiniciar)
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((10, d_k))

    # Puntajes de compatibilidad (10,) = K @ q
    puntajes = K @ q

    # Distribuciones de atención
    w_sin = softmax_estable(puntajes)                 # sin escalar
    w_esc = softmax_estable(puntajes / np.sqrt(d_k))  # escalado por sqrt(d_k)

    # Cifras reportadas (precisión completa)
    pm_sin = float(np.max(w_sin))
    h_sin = entropia_bits(w_sin)
    pm_esc = float(np.max(w_esc))
    h_esc = entropia_bits(w_esc)

    peso_maximo_sin_escalar_por_dk[str(d_k)] = pm_sin
    entropia_bits_sin_escalar_por_dk[str(d_k)] = h_sin
    peso_maximo_escalado_por_dk[str(d_k)] = pm_esc
    entropia_bits_escalado_por_dk[str(d_k)] = h_esc

    tabla_filas.append({
        "d_k": int(d_k),
        "peso_maximo_sin_escalar": f"{pm_sin:.4f}",
        "entropia_bits_sin_escalar": f"{h_sin:.4f}",
        "peso_maximo_escalado": f"{pm_esc:.4f}",
        "entropia_bits_escalado": f"{h_esc:.4f}",
    })

# ------------------------------------------------------------------
# Guardar TODAS las cifras en resultados.json (precisión completa)
# ------------------------------------------------------------------
resultados = {
    "peso_maximo_sin_escalar_por_dk": peso_maximo_sin_escalar_por_dk,
    "entropia_bits_sin_escalar_por_dk": entropia_bits_sin_escalar_por_dk,
    "peso_maximo_escalado_por_dk": peso_maximo_escalado_por_dk,
    "entropia_bits_escalado_por_dk": entropia_bits_escalado_por_dk,
    "tabla_4_decimales": tabla_filas,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# Impresión: tabla con cuatro decimales
# ------------------------------------------------------------------
encabezado = (
    f"{'d_k':>6} | {'peso_max_sin':>14} | {'H_bits_sin':>14} | "
    f"{'peso_max_esc':>14} | {'H_bits_esc':>14}"
)
print("Efecto de escalar por sqrt(d_k)  (rng = np.random.default_rng(0), 10 claves)")
print(encabezado)
print("-" * len(encabezado))
for fila in tabla_filas:
    print(
        f"{fila['d_k']:>6} | "
        f"{fila['peso_maximo_sin_escalar']:>14} | "
        f"{fila['entropia_bits_sin_escalar']:>14} | "
        f"{fila['peso_maximo_escalado']:>14} | "
        f"{fila['entropia_bits_escalado']:>14}"
    )

# Cifras principales con precisión completa
print("\nCifras completas (sin redondear):")
for d_k in d_k_lista:
    k = str(d_k)
    print(
        f"d_k={d_k}: sin escalar -> peso_max={peso_maximo_sin_escalar_por_dk[k]!r}, "
        f"entropia_bits={entropia_bits_sin_escalar_por_dk[k]!r} | "
        f"escalado -> peso_max={peso_maximo_escalado_por_dk[k]!r}, "
        f"entropia_bits={entropia_bits_escalado_por_dk[k]!r}"
    )

print("\nResultados guardados en resultados.json")
