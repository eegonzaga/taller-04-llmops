import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Partición de T1 (75/25 estratificada, random_state=7)
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)
n_train = len(X_train)

ms = [20, 50, 100, 200, 400]
acc_gnb, acc_lr = {}, {}

for m in ms + [n_train]:
    if m == n_train:
        Xs, ys = X_train, y_train
        etiqueta = "completo"
    else:
        Xs, _, ys, _ = train_test_split(
            X_train, y_train, train_size=m, stratify=y_train, random_state=7
        )
        etiqueta = str(m)

    # Generativo: GaussianNB sobre atributos originales, sin escalar
    gnb = GaussianNB().fit(Xs, ys)
    acc_gnb[etiqueta] = float(accuracy_score(y_test, gnb.predict(X_test)))

    # Discriminativo: LR sobre atributos estandarizados (scaler ajustado solo con Xs)
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xs), ys)
    acc_lr[etiqueta] = float(accuracy_score(y_test, lr.predict(sc.transform(X_test))))

print("Exactitud GaussianNB por m:", acc_gnb)
print("Exactitud Regresión Logística por m:", acc_lr)

# Figura: curva de aprendizaje
xs = ms + [n_train]
def serie(d):
    return [d[str(m)] if m != n_train else d["completo"] for m in xs]

plt.figure(figsize=(7, 5))
plt.plot(xs, serie(acc_gnb), "o-", label="GaussianNB (generativo)")
plt.plot(xs, serie(acc_lr), "s-", label="Regresión logística (discriminativo)")
plt.xscale("log")
plt.xticks(xs, [str(v) for v in xs])
plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Exactitud en el conjunto de prueba")
plt.title("Curva de aprendizaje: Naive Bayes vs. Regresión logística (dígitos)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=150)

resultados = {
    "accuracy_gnb_por_m": acc_gnb,
    "accuracy_lr_por_m": acc_lr,
    "figura_curva_aprendizaje": "curva_aprendizaje.png",
    "tam_train_completo": int(n_train),
}
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)
