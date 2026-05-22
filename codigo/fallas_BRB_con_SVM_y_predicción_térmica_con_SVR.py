# Diagnóstico académico de fallas BRB con SVM + predicción térmica con SVR

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)

np.random.seed(42)
os.makedirs("figuras", exist_ok=True)

# ============================================================
# 1. Dataset sintético
# ============================================================

n_samples = 300
f_sync = 1800  # rpm, motor 4 polos a 60 Hz

def generar_estado(estado, n):
    if estado == "Normal":
        carga = np.random.normal(55, 12, n)
        slip = np.random.normal(0.025, 0.008, n)
        ia = np.random.normal(8.0, 0.7, n)
        ib = np.random.normal(8.1, 0.7, n)
        ic = np.random.normal(7.9, 0.7, n)
        vibracion = np.random.normal(1.8, 0.35, n)

    elif estado == "Sobrecarga":
        carga = np.random.normal(92, 10, n)
        slip = np.random.normal(0.055, 0.012, n)
        ia = np.random.normal(11.8, 1.0, n)
        ib = np.random.normal(12.0, 1.0, n)
        ic = np.random.normal(11.7, 1.0, n)
        vibracion = np.random.normal(2.7, 0.55, n)

    elif estado == "Barras rotas":
        carga = np.random.normal(72, 13, n)
        slip = np.random.normal(0.073, 0.016, n)
        ia = np.random.normal(10.7, 1.1, n)
        ib = np.random.normal(9.3, 1.1, n)
        ic = np.random.normal(11.5, 1.2, n)
        vibracion = np.random.normal(4.0, 0.8, n)

    velocidad = f_sync * (1 - slip)

    # Temperaturas simuladas con relación física aproximada
    corriente_prom = (ia + ib + ic) / 3
    temp_estator = 35 + 3.8*corriente_prom + 0.18*carga + np.random.normal(0, 4, n)
    temp_rotor = temp_estator + 8 + 180*slip + 2.8*vibracion + np.random.normal(0, 5, n)

    return pd.DataFrame({
        "corriente_A": ia,
        "corriente_B": ib,
        "corriente_C": ic,
        "temp_estator": temp_estator,
        "temp_rotor": temp_rotor,
        "carga_mecanica": carga,
        "velocidad": velocidad,
        "vibracion": vibracion,
        "deslizamiento": slip,
        "estado": estado
    })

df = pd.concat([
    generar_estado("Normal", n_samples),
    generar_estado("Sobrecarga", n_samples),
    generar_estado("Barras rotas", n_samples)
], ignore_index=True)

df.to_csv("dataset_motor_svm.csv", index=False)

# ============================================================
# 2. Clasificación SVM
# ============================================================

X = df.drop(columns=["estado"])
y = df["estado"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

modelo_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=5, gamma="scale"))
])

modelo_svm.fit(X_train, y_train)
y_pred = modelo_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nMétricas del clasificador SVM")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=modelo_svm.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=modelo_svm.classes_)
disp.plot(values_format="d")
plt.title("Matriz de confusión - SVM")
plt.tight_layout()
plt.savefig("figuras/matriz_confusion_svm.png", dpi=300)
plt.close()

# ============================================================
# 3. Predicción térmica con SVR
# ============================================================

X_temp = df.drop(columns=["temp_rotor", "estado"])
y_temp = df["temp_rotor"]

Xtr, Xte, ytr, yte = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

modelo_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=50, gamma="scale", epsilon=2.0))
])

modelo_svr.fit(Xtr, ytr)
ytemp_pred = modelo_svr.predict(Xte)

mae = mean_absolute_error(yte, ytemp_pred)
rmse = np.sqrt(mean_squared_error(yte, ytemp_pred))
r2 = r2_score(yte, ytemp_pred)

print("\nMétricas de predicción térmica SVR")
print(f"MAE : {mae:.4f} °C")
print(f"RMSE: {rmse:.4f} °C")
print(f"R²  : {r2:.4f}")

# Temperatura real vs predicha
plt.figure(figsize=(7, 5))
plt.scatter(yte, ytemp_pred, alpha=0.7)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], linestyle="--")
plt.xlabel("Temperatura real del rotor [°C]")
plt.ylabel("Temperatura predicha del rotor [°C]")
plt.title("Temperatura real vs predicha - SVR")
plt.tight_layout()
plt.savefig("figuras/temp_real_vs_predicha.png", dpi=300)
plt.close()

# ============================================================
# 4. Gráficas comparativas
# ============================================================

# Temperatura del rotor por estado
plt.figure(figsize=(7, 5))
for estado in df["estado"].unique():
    datos = df[df["estado"] == estado]["temp_rotor"]
    plt.hist(datos, alpha=0.5, bins=20, label=estado)

plt.xlabel("Temperatura del rotor [°C]")
plt.ylabel("Frecuencia")
plt.title("Distribución de temperatura del rotor por estado")
plt.legend()
plt.tight_layout()
plt.savefig("figuras/temp_rotor_estado.png", dpi=300)
plt.close()

# Vibración vs deslizamiento
plt.figure(figsize=(7, 5))
for estado in df["estado"].unique():
    datos = df[df["estado"] == estado]
    plt.scatter(datos["deslizamiento"], datos["vibracion"], alpha=0.7, label=estado)

plt.xlabel("Deslizamiento [p.u.]")
plt.ylabel("Vibración RMS [mm/s]")
plt.title("Vibración vs deslizamiento")
plt.legend()
plt.tight_layout()
plt.savefig("figuras/vibracion_deslizamiento.png", dpi=300)
plt.close()

# Corrientes promedio por estado
df_group = df.groupby("estado")[["corriente_A", "corriente_B", "corriente_C"]].mean()
df_group.plot(kind="bar", figsize=(7, 5))
plt.ylabel("Corriente promedio [A]")
plt.title("Corriente promedio por fase según el estado")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figuras/corrientes_estado.png", dpi=300)
plt.close()

# Guardar métricas
metricas = pd.DataFrame({
    "Metrica": ["Accuracy", "Precision", "Recall", "F1-score", "MAE_temp", "RMSE_temp", "R2_temp"],
    "Valor": [accuracy, precision, recall, f1, mae, rmse, r2]
})

metricas.to_csv("metricas_modelo.csv", index=False)

print("\nArchivos generados:")
print("- dataset_motor_svm.csv")
print("- metricas_modelo.csv")
print("- figuras/matriz_confusion_svm.png")
print("- figuras/temp_real_vs_predicha.png")
print("- figuras/temp_rotor_estado.png")
print("- figuras/vibracion_deslizamiento.png")
print("- figuras/corrientes_estado.png")