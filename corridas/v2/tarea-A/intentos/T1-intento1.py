import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Cargar dígitos
digits = load_digits()
X, y = digits.data, digits.target

# Partición única: 75/25 estratificada, random_state=7
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)

# Cifras
resultados = {
    "n_ejemplos": int(X.shape[0]),
    "n_atributos": int(X.shape[1]),
    "n_clases": int(len(np.unique(y))),
    "tam_train": int(X_train.shape[0]),
    "tam_test": int(X_test.shape[0]),
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# Guardar la partición para las subtareas siguientes
np.savez("datos_particion.npz", X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)

print("Cifras de la Parte 1:")
for k, v in resultados.items():
    print(f"  {k}: {v}")
