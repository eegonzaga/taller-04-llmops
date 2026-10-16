import json
import numpy as np

# Matrices del enunciado
Q = np.array([[1.0, 0.0, 1.0, 0.5],
              [0.0, 2.0, 0.0, 1.0]])
K = np.array([[1.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 1.0, 0.5],
              [1.0, 0.0, 1.0, 1.0]])
V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [2.0, 2.0]])

d_k = Q.shape[1]

# Puntuaciones escaladas
scores = Q @ K.T / np.sqrt(d_k)

# Softmax numéricamente estable (restar el máximo de cada fila)
scores_max = scores.max(axis=1, keepdims=True)
exp_scores = np.exp(scores - scores_max)
pesos = exp_scores / exp_scores.sum(axis=1, keepdims=True)

# Salida
salida = pesos @ V

# Guardar resultados con precisión completa
resultados = {
    "pesos_atencion_2x3": pesos.tolist(),
    "salida_atencion_2x2": salida.tolist(),
    "scores_escalados_2x3": scores.tolist(),
    "d_k": int(d_k),
}
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# Imprimir con cuatro decimales
np.set_printoptions(precision=4, suppress=True)
print("Scores escalados QK^T/sqrt(d_k):")
print(scores)
print("\nPesos de atencion (2x3):")
print(pesos)
print("\nSalida de atencion (2x2):")
print(salida)
