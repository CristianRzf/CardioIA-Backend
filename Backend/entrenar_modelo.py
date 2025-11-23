import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# --- 1. Cargar el dataset ---
try:
    df = pd.read_csv('heart_disease_dataset.csv')
    print("✅ Dataset cargado correctamente.")
except FileNotFoundError:
    raise FileNotFoundError("❌ No se encontró el archivo 'heart_disease_dataset.csv'. "
                            "Asegúrate de que esté en la misma carpeta que este script.")


# --- 2. Limpieza y Preprocesamiento ---

# Rellenar valores faltantes en 'Alcohol Intake' con la moda
if 'Alcohol Intake' in df.columns:
    df['Alcohol Intake'].fillna(df['Alcohol Intake'].mode()[0], inplace=True)

# Variables categóricas a codificar
categorical_cols = [
    'Gender', 'Smoking', 'Alcohol Intake', 'Exercise Hours', 'Family History',
    'Diabetes', 'Obesity', 'Stress Level', 'Exercise Induced Angina', 'Chest Pain Type'
]

# One-hot encoding (crea columnas binarias para categorías)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separar características (X) y variable objetivo (y)
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# --- 3. División de datos ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Datos divididos correctamente: {X_train.shape[0]} entrenamiento / {X_test.shape[0]} prueba")

# --- 4. Escalado de variables numéricas ---
numerical_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Blood Sugar']

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# --- 5. Entrenamiento del modelo ---
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("🤖 Modelo de Regresión Logística entrenado correctamente.")

# --- 6. Evaluación del modelo ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n🎯 Accuracy del modelo: {accuracy:.4f}")
print("\n📋 Reporte de Clasificación:")
print(classification_rep)
print("\n🧩 Matriz de Confusión:")
print(pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

# --- 7. Guardar artefactos para el backend ---
joblib.dump(model, 'cardio_model.pkl')
joblib.dump(scaler, 'cardio_scaler.pkl')

train_columns = X_train.columns.tolist()
with open('cardio_columns.json', 'w') as f:
    json.dump(train_columns, f)

print("\n✅ Archivos generados correctamente:")
print(" - Modelo guardado en: cardio_model.pkl")
print(" - Scaler guardado en: cardio_scaler.pkl")
print(" - Columnas guardadas en: cardio_columns.json")
