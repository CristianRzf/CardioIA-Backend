import joblib
import json
import pandas as pd
import os  # <-- Importante para variables de entorno
from dotenv import load_dotenv  # <-- Importante
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore

# --- 0. Cargar variables de entorno (.env) ---
# Esto carga las claves desde el archivo .env cuando estás en local
# Truco para encontrar archivos en la misma carpeta que main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Construimos la ruta completa
    model_path = os.path.join(BASE_DIR, 'cardio_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'cardio_scaler.pkl')
    columns_path = os.path.join(BASE_DIR, 'cardio_columns.json')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(columns_path, 'r') as f:
        train_columns = json.load(f)
    print(f"--- ✅ Backend: Modelos cargados desde {BASE_DIR} ---")
except Exception as e:
    print(f"--- ❌ Error cargando modelos: {e}")
    print("Asegúrate de que .pkl y .json estén en la misma carpeta.")
    model, scaler, train_columns = None, None, None

# ---------------------------------------------
# 1.5. Conexión SEGURA a Firebase Firestore
# ---------------------------------------------
db = None
try:
    # INTENTO 1: Buscar el contenido JSON en una variable de entorno (Para Producción/Render)
    # (En Render crearás una variable llamada FIREBASE_CREDENTIALS_JSON con todo el contenido del archivo pegado)
    firebase_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")

    # INTENTO 2: Buscar el archivo local (Para tu PC)
    firebase_file_path = os.getenv("FIREBASE_CREDENTIALS_FILE")

    if firebase_json_str:
        # Si estamos en producción y tenemos el JSON como texto
        cred_dict = json.loads(firebase_json_str)
        cred = credentials.Certificate(cred_dict)
    elif firebase_file_path and os.path.exists(firebase_file_path):
        # Si estamos en local y tenemos el archivo
        cred = credentials.Certificate(firebase_file_path)
    else:
        raise Exception("No se encontraron credenciales de Firebase (ni variable de entorno ni archivo local).")

    # Inicializar solo si no está inicializado ya
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    print("--- ✅ Backend: Conexión a Firebase Firestore establecida. ---")

except Exception as e:
    print(f"--- ❌ ERROR CRÍTICO AL CONECTAR A FIREBASE: {e} ---")
    db = None

# ---------------------------------------------
# 1.6. Configuración de la API de Gemini
# ---------------------------------------------
client = None

# Leemos la clave de la variable de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("--- ✅ Backend: Cliente Gemini inicializado. ---")
    else:
        print("--- ⚠️ WARNING: No se encontró la GEMINI_API_KEY en las variables de entorno. ---")
except Exception as e:
    print(f"--- ❌ ERROR CRÍTICO AL CONECTAR A GEMINI: {e} ---")
    client = None


# --- 2. Modelos de datos ---
class HeartData(BaseModel):
    edad: int = Field(..., gt=17, lt=121)
    sexo: int = Field(..., ge=0, le=1)
    colesterol: int = Field(..., gt=50, lt=600)
    presion_arterial: int = Field(..., gt=50, lt=300)
    frecuencia_cardiaca: int = Field(..., gt=40, lt=250)
    fumador: int = Field(..., ge=0, le=1)
    consumo_alcohol: int = Field(..., ge=0, le=1)
    horas_ejercicio: int = Field(..., ge=0, lt=100)
    historial_familiar: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    obesidad: int = Field(..., ge=0, le=1)
    nivel_estres: int = Field(..., ge=1, le=10)
    nivel_azucar: int = Field(..., gt=30, lt=600)
    angina_inducida_ejercicio: int = Field(..., ge=0, le=1)
    tipo_dolor_pecho: int = Field(..., ge=0, le=3)


class PredictionResponse(BaseModel):
    probabilidad: float
    nivel_riesgo: str
    factores_influyentes: list[str]
    reporte_ia: str


# --- 3. Inicializar FastAPI ---
app = FastAPI(title="CardioIA API")

# --- 4. CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- FUNCION DE GENERACIÓN DE REPORTE CON GEMINI ---
async def generate_detailed_report(data: dict, prob_porcentaje: float, nivel: str) -> str:
    if not client:
        return "Error: Servicio de IA no disponible (API Key no configurada)."

    datos_paciente_str = "\n".join([
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in data.items()
        if k not in ['angina_inducida_ejercicio', 'tipo_dolor_pecho']
    ])

    prompt = f"""
    Eres un asistente médico experto en Cardiología para CardioIA.
    Genera un reporte breve, profesional y motivacional.

    **Datos del Paciente:**
    Riesgo: {nivel} ({prob_porcentaje}%)

    {datos_paciente_str}

    Estructura:
    1. Resumen de Riesgo
    2. Análisis de Factores
    3. Recomendaciones

    Formato Markdown.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"--- ❌ ERROR GEMINI: {e} ---")
        return "No se pudo generar el reporte detallado por el momento."


# --- 5. Preprocesamiento ---
def preprocess_data(data: HeartData) -> pd.DataFrame:
    # Mapeos (igual que antes)
    gender_map = {1: 'Male', 0: 'Female'}
    smoking_map = {1: 'Current', 0: 'Never'}
    alcohol_map = {1: 'Heavy', 0: 'None'}
    family_history_map = {1: 'Yes', 0: 'No'}
    diabetes_map = {1: 'Yes', 0: 'No'}
    obesity_map = {1: 'Yes', 0: 'No'}
    angina_map = {1: 'Yes', 0: 'No'}
    chest_pain_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}

    raw_data = {
        'Age': data.edad,
        'Gender': gender_map.get(data.sexo),
        'Cholesterol': data.colesterol,
        'Blood Pressure': data.presion_arterial,
        'Heart Rate': data.frecuencia_cardiaca,
        'Smoking': smoking_map.get(data.fumador),
        'Alcohol Intake': alcohol_map.get(data.consumo_alcohol),
        'Exercise Hours': str(data.horas_ejercicio),
        'Family History': family_history_map.get(data.historial_familiar),
        'Diabetes': diabetes_map.get(data.diabetes),
        'Obesity': obesity_map.get(data.obesidad),
        'Stress Level': str(data.nivel_estres),
        'Blood Sugar': data.nivel_azucar,
        'Exercise Induced Angina': angina_map.get(data.angina_inducida_ejercicio),
        'Chest Pain Type': chest_pain_map.get(data.tipo_dolor_pecho)
    }

    df = pd.DataFrame([raw_data])
    categorical_cols = ['Gender', 'Smoking', 'Alcohol Intake', 'Exercise Hours', 'Family History', 'Diabetes',
                        'Obesity', 'Stress Level', 'Exercise Induced Angina', 'Chest Pain Type']
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_reindexed = df_processed.reindex(columns=train_columns, fill_value=0)
    numerical_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Blood Sugar']
    df_reindexed[numerical_cols] = scaler.transform(df_reindexed[numerical_cols])

    return df_reindexed


# --- 6. Endpoint de predicción ---
@app.post("/api/predict", response_model=PredictionResponse)  # Asegúrate de que la ruta sea /api/predict
async def predict_heart_disease(data: HeartData):
    if not all([model, scaler, train_columns]):
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    try:
        processed_df = preprocess_data(data)
        probabilidad = model.predict_proba(processed_df)[0][1]
        prob_porcentaje = round(float(probabilidad) * 100, 2)

        if prob_porcentaje >= 65:
            nivel = "Alto"
        elif prob_porcentaje >= 35:
            nivel = "Moderado"
        else:
            nivel = "Bajo"

        factores = ["Análisis de factores pendiente."]

        # Generar reporte
        reporte_ia = await generate_detailed_report(data.dict(), prob_porcentaje, nivel)

        # Guardar en Firestore
        if db:
            try:
                timestamp = datetime.now().isoformat()
                clinical_ref = db.collection('ClinicalData').document()
                datos_guardar = data.dict()
                datos_guardar['timestamp'] = timestamp
                clinical_ref.set(datos_guardar)

                db.collection('Evaluations').add({
                    "prediction_id": clinical_ref.id,
                    "probabilidad_riesgo": prob_porcentaje,
                    "nivel_riesgo": nivel,
                    "reporte_ia": reporte_ia,
                    "timestamp": timestamp,
                })
            except Exception as e:
                print(f"Error guardando en Firebase: {e}")

        return {
            "probabilidad": prob_porcentaje,
            "nivel_riesgo": nivel,
            "factores_influyentes": factores,
            "reporte_ia": reporte_ia
        }

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.get("/")
def read_root():
    return {"message": "CardioIA Backend API está funcionando correctamente 🚀"}