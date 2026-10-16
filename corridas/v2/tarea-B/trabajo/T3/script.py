import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- Codificación posicional sinusoidal (d_model=16, pos 0..49) ----------
d_model = 16
n_pos = 50
pos = np.arange(n_pos)                      # posiciones 0..49
i2 = np.arange(0, d_model, 2)               # exponentes 2i = 0,2,4,...,14
angle_rates = 1.0 / (10000 ** (i2 / d_model))   # 1/10000^(2i/d_model)
angles = pos[:, None] * angle_rates[None, :]    # (50, 8)

PE = np.zeros((n_pos, d_model))
PE[:, 0::2] = np.sin(angles)   # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE[:, 1::2] = np.cos(angles)   # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

PE_10_0 = float(PE[10, 0])
PE_10_1 = float(PE[10, 1])
PE_25_6 = float(PE[25, 6])

# ---------- Atención escalada (misma función estable de la Parte 1) ----------
def softmax_estable(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def atencion(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    return softmax_estable(scores), scores

# PE[10] como consulta (1x16), matriz PE completa como claves y valores (50x16)
Q = PE[10:11, :]
pesos, scores = atencion(Q, PE, PE)         # pesos: (1, 50)
posicion_mayor_peso = int(np.argmax(pesos[0]))
valor_mayor_peso = float(pesos[0, posicion_mayor_peso])

# ---------- Guardar resultados ----------
resultados = {
    "PE_10_0": PE_10_0,
    "PE_10_1": PE_10_1,
    "PE_25_6": PE_25_6,
    "posicion_mayor_peso": posicion_mayor_peso,
    "valor_mayor_peso": valor_mayor_peso,
}
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("PE[10, 0] =", PE_10_0)
print("PE[10, 1] =", PE_10_1)
print("PE[25, 6] =", PE_25_6)
print("Posición con mayor peso de atención:", posicion_mayor_peso)
print("Valor del mayor peso:", valor_mayor_peso)

# ---------- Figura: mapa de calor de PE + pesos de atención ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                               gridspec_kw={"width_ratios": [2, 1]})
im = ax1.imshow(PE, aspect="auto", cmap="RdBu_r")
fig.colorbar(im, ax=ax1)
ax1.set_xlabel("Dimensión (i)")
ax1.set_ylabel("Posición (pos)")
ax1.set_title("Codificación posicional sinusoidal (d_model=16, pos 0-49)")

ax2.plot(np.arange(n_pos), pesos[0], lw=1)
ax2.scatter([posicion_mayor_peso], [valor_mayor_peso], color="red", zorder=3,
            label=f"máx: pos {posicion_mayor_peso} = {valor_mayor_peso:.4f}")
ax2.set_xlabel("Posición (clave)")
ax2.set_ylabel("Peso de atención")
ax2.set_title("Atención: consulta = PE[10]")
ax2.legend()

plt.tight_layout()
plt.savefig("pe_heatmap.png", dpi=150)
