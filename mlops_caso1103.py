import os
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pycaret.classification import setup, compare_models, finalize_model, save_model, load_model

# Configuración de FastAPI
app = FastAPI()

# Ruta donde se guarda el modelo
MODEL_PATH = "ml_model"

# 1. Función para entrenar y guardar el modelo
@app.get("/train")
def train_model():
    df = pd.read_csv("data.csv")  # Se asume un dataset llamado data.csv en la raíz
    setup(df, target='target', silent=True, session_id=123)
    best_model = compare_models()
    final_model = finalize_model(best_model)
    save_model(final_model, MODEL_PATH)
    return {"message": "Modelo entrenado y guardado"}

# 2. Cargar el modelo entrenado
@app.on_event("startup")
def load_trained_model():
    global model
    if os.path.exists(f"{MODEL_PATH}.pkl"):
        model = load_model(MODEL_PATH)
    else:
        model = None

# 3. Endpoint de predicción
@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Modelo no está entrenado"}
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}

# 4. Ejecutar API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
