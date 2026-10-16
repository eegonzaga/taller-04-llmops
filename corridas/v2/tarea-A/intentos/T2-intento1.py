import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Datos y partición fija (75/25, estratificada, random_state=7)
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)

# GaussianNB sobre atributos originales sin escalar
gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred_gnb = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, pred_gnb)
f1_gnb = f1_score(y_test, pred_gnb, average="macro")

# Regresión logística sobre atributos estandarizados (scaler ajustado solo con train)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
lr = LogisticRegression(max_iter=3000)
lr.fit(X_train_s, y_train)
pred_lr = lr.predict(X_test_s)
acc_lr = accuracy_score(y_test, pred_lr)
f1_lr = f1_score(y_test, pred_lr, average="macro")

# Guardar resultados
resultados = {
    "accuracy_gnb": float(acc_gnb),
    "f1_macro_gnb": float(f1_gnb),
    "accuracy_lr": float(acc_lr),
    "f1_macro_lr": float(f1_lr),
}
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# Tabla en consola
print(f"{'Modelo':<20}{'Accuracy':>12}{'F1 macro':>12}")
print(f"{'GaussianNB':<20}{acc_gnb:>12.4f}{f1_gnb:>12.4f}")
print(f"{'LogisticRegression':<20}{acc_lr:>12.4f}{f1_lr:>12.4f}")
