# -*- coding: utf-8 -*-
"""
T3 - Parte 3 - Codificación posicional sinusoidal (Vaswani et al., 2017).

1) Calcula PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   con d_model = 16 y pos = 0..49.
2) Reporta PE[10, 0], PE[10, 1] y PE[25, 6].
3) Dibuja la matriz 50x16 como mapa de calor (PNG).
4) Reutiliza la función de atención escalada de la Parte 1
   (softmax numéricamente estable, resta el máximo de cada fila) con
   consulta = PE[10] y claves/valores = matriz PE completa; identifica
   la posición que recibe el mayor peso y cuánto vale.

Salidas: resultados.json, T3_parte3_mapa_calor_PE.png,
         T3_parte3_pesos_atencion_PE10.png
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------
# 1) Función de atención escalada de la Parte 1 (se reutiliza aquí)
# ------------------------------------------------------------------
def softmax_estable(x, axis=-1):
    """Softmax numéricamente estable: resta el máximo de cada fila."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def atencion_escalada(Q, K, V):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Q: (n_q, d_k) | K: (n_k, d_k) | V: (n_k, d_v)
    Devuelve (puntajes, escalados, pesos, salida).
    """
    d_k = Q.shape[-1]
    puntajes = Q @ K.T                      # (n_q, n_k)
    escalados = puntajes / np.sqrt(d_k)     # (n_q, n_k)
    pesos = softmax_estable(escalados, axis=-1)
    salida = pesos @ V                      # (n_q, d_v)
    return puntajes, escalados, pesos, salida


# ------------------------------------------------------------------
# 2) Verificación: la misma función reproduce los resultados de T1
# ------------------------------------------------------------------
verif = {"T1_cargado": False}
ruta_t1 = os.path.join("entradas", "T1.json")
if os.path.exists(ruta_t1):
    try:
        with open(ruta_t1, "r", encoding="utf-8") as f:
            t1 = json.load(f)
        Q1 = np.asarray(t1["Q_2x4"], dtype=float)
        K1 = np.asarray(t1["K_3x4"], dtype=float)
        V1 = np.asarray(t1["V_3x2"], dtype=float)
        _, _, w1, s1 = atencion_escalada(Q1, K1, V1)
        w1_ref = np.asarray(t1["pesos_atencion_2x3"], dtype=float)
        s1_ref = np.asarray(t1["salida_atencion_2x2"], dtype=float)
        verif = {
            "T1_cargado": True,
            "d_k_T1": t1.get("d_k"),
            "max_abs_diff_pesos": float(np.max(np.abs(w1 - w1_ref))),
            "max_abs_diff_salida": float(np.max(np.abs(s1 - s1_ref))),
        }
        print("[Reutilización] Función de la Parte 1 verificada contra entradas/T1.json "
              f"(max diff pesos = {verif['max_abs_diff_pesos']:.2e}, "
              f"max diff salida = {verif['max_abs_diff_salida']:.2e})")
    except Exception as exc:  # un problema con T1 no debe interrumpir la subtarea
        verif = {"T1_cargado": False, "error": str(exc)}
        print("[Aviso] No se pudo verificar contra T1.json:", exc)

# ------------------------------------------------------------------
# 3) Codificación posicional sinusoidal (d_model = 16, pos = 0..49)
# ------------------------------------------------------------------
d_model = 16
n_pos = 50  # posiciones 0 a 49

pos = np.arange(n_pos, dtype=float)                 # (50,)
jdims = np.arange(d_model, dtype=float)             # dimensiones 0..15

# Exponente 2i/d_model de la fórmula del paper:
#   - dimensión par   j = 2i   -> exponente  j / d_model
#   - dimensión impar j = 2i+1 -> exponente (j-1) / d_model
exponente = np.where(jdims % 2 == 0, jdims, jdims - 1.0) / d_model   # (16,)
divisor = np.power(10000.0, exponente)                               # (16,)

angulos = pos[:, None] / divisor[None, :]                            # (50, 16)
PE = np.zeros((n_pos, d_model), dtype=float)
PE[:, 0::2] = np.sin(angulos[:, 0::2])   # dimensiones pares:  sin
PE[:, 1::2] = np.cos(angulos[:, 1::2])   # dimensiones impares: cos

PE_10_0 = float(PE[10, 0])
PE_10_1 = float(PE[10, 1])
PE_25_6 = float(PE[25, 6])

print("\nd_model =", d_model, "| posiciones = 0 ..", n_pos - 1)
print(f"PE[10, 0] = sin(10 / 10000^(0/16))  = {PE_10_0:.10f}")
print(f"PE[10, 1] = cos(10 / 10000^(0/16))  = {PE_10_1:.10f}")
print(f"PE[25, 6] = sin(25 / 10000^(6/16))  = {PE_25_6:.10f}")

# ------------------------------------------------------------------
# 4) Atención: consulta = PE[10], claves y valores = PE completa
# ------------------------------------------------------------------
consulta = PE[10:11, :]        # (1, 16)  -> aquí d_k = 16
claves = PE                    # (50, 16)
valores = PE                   # (50, 16)

puntajes, escalados, pesos, salida = atencion_escalada(consulta, claves, valores)
pesos_vec = pesos[0]           # (50,)
salida_vec = salida[0]         # (16,)

pos_mayor = int(np.argmax(pesos_vec))
valor_mayor = float(pesos_vec[pos_mayor])

print("\nAtención con consulta PE[10] (d_k = 16, sqrt(d_k) = %.1f):"
      % np.sqrt(consulta.shape[-1]))
orden = np.argsort(pesos_vec)[::-1][:5]
for p in orden:
    print(f"  pos {int(p):2d} -> peso = {pesos_vec[p]:.8f}")
print(f"Posición con mayor peso: {pos_mayor}")
print(f"Valor del mayor peso:    {valor_mayor:.10f}")
print(f"Suma de los pesos (debe ser 1): {float(np.sum(pesos_vec)):.10f}")

# ------------------------------------------------------------------
# 5) Figura 1: mapa de calor de la matriz PE (50 x 16)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(PE, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
               interpolation="nearest", origin="upper")
ax.set_xlabel("Dimensión j (0-15)")
ax.set_ylabel("Posición pos (0-49)")
ax.set_title("Codificación posicional sinusoidal — d_model = 16, pos = 0..49\n"
             "PE(pos,2i)=sin(pos/10000^(2i/d_model)),  PE(pos,2i+1)=cos(pos/10000^(2i/d_model))")
ax.set_xticks(np.arange(0, d_model, 1))
ax.set_yticks(np.arange(0, n_pos, 5))
for (p, jj) in [(10, 0), (10, 1), (25, 6)]:   # marcar las celdas reportadas
    ax.add_patch(Rectangle((jj - 0.5, p - 0.5), 1, 1, fill=False,
                           edgecolor="black", linewidth=1.8))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Valor de PE")
fig.tight_layout()
fig.savefig("T3_parte3_mapa_calor_PE.png", dpi=150)
plt.close(fig)
print("\nFigura guardada: T3_parte3_mapa_calor_PE.png")

# ------------------------------------------------------------------
# 6) Figura 2: pesos de atención de la consulta PE[10] sobre las 50 posiciones
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
colores = ["crimson" if p == pos_mayor else "steelblue" for p in range(n_pos)]
ax.bar(np.arange(n_pos), pesos_vec, color=colores)
ax.set_xlabel("Posición (clave/valor)")
ax.set_ylabel("Peso de atención")
ax.set_title(f"Consulta = PE[10], claves/valores = PE (pos 0..49) — "
             f"mayor peso: pos {pos_mayor} = {valor_mayor:.6f}")
ax.set_xticks(np.arange(0, n_pos, 5))
fig.tight_layout()
fig.savefig("T3_parte3_pesos_atencion_PE10.png", dpi=150)
plt.close(fig)
print("Figura guardada: T3_parte3_pesos_atencion_PE10.png")

# ------------------------------------------------------------------
# 7) Guardar TODAS las cifras en resultados.json
# ------------------------------------------------------------------
resultados = {
    "d_model": int(d_model),
    "n_posiciones": int(n_pos),
    "PE_10_0": PE_10_0,
    "PE_10_1": PE_10_1,
    "PE_25_6": PE_25_6,
    "posicion_mayor_peso": int(pos_mayor),
    "valor_mayor_peso": valor_mayor,
    "d_k_atencion_PE": int(consulta.shape[-1]),
    "sqrt_d_k_atencion_PE": float(np.sqrt(consulta.shape[-1])),
    "PE_50x16": PE.tolist(),
    "puntajes_escalados_PE10_1x50": escalados[0].tolist(),
    "pesos_atencion_PE10_1x50": pesos_vec.tolist(),
    "salida_atencion_PE10_1x16": salida_vec.tolist(),
    "suma_pesos_atencion": float(np.sum(pesos_vec)),
    "verificacion_funcion_T1": verif,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("Resultados guardados en resultados.json")
