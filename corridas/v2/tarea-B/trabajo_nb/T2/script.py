import json
import numpy as np

rng = np.random.default_rng(0)

resultados = {
    "peso_maximo_sin_escalar": {},
    "entropia_bits_sin_escalar": {},
    "peso_maximo_escalado": {},
    "entropia_bits_escalado": {},
}

filas = []
for d_k in [4, 64, 512]:
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((10, d_k))

    puntajes = K @ q

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    p_sin = softmax(puntajes)
    p_esc = softmax(puntajes / np.sqrt(d_k))

    ent_sin = float(-np.sum(p_sin * np.log2(p_sin)))
    ent_esc = float(-np.sum(p_esc * np.log2(p_esc)))

    pm_sin = float(p_sin.max())
    pm_esc = float(p_esc.max())

    resultados["peso_maximo_sin_escalar"][str(d_k)] = pm_sin
    resultados["entropia_bits_sin_escalar"][str(d_k)] = ent_sin
    resultados["peso_maximo_escalado"][str(d_k)] = pm_esc
    resultados["entropia_bits_escalado"][str(d_k)] = ent_esc

    filas.append((d_k, pm_sin, ent_sin, pm_esc, ent_esc))

print(f"{'d_k':>5} | {'peso_max_sin':>13} | {'entropia_sin(bits)':>19} | {'peso_max_esc':>13} | {'entropia_esc(bits)':>19}")
for d_k, pm_sin, ent_sin, pm_esc, ent_esc in filas:
    print(f"{d_k:>5} | {pm_sin:>13.4f} | {ent_sin:>19.4f} | {pm_esc:>13.4f} | {ent_esc:>19.4f}")

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)
