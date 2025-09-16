""" API para el modelo de clasificación de EEG."""
import re
import pickle
import uuid
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
import mne
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel, Field

class FeatureInput(BaseModel):
    Delta: float = Field(..., example=1.2)
    Theta: float = Field(..., example=0.8)
    Alpha: float = Field(..., example=0.5)
    Beta: float = Field(..., example=0.3)
    Gamma: float = Field(..., example=0.1)

class PredictionOutput(BaseModel):
    prediction: str
    probabilities: list[float]

class ExplainOutput(BaseModel):
    explanation: dict | str  # HTML string or dict

# ======================
# 1. Cargar modelo y data de entrenamiento
# ======================
PATH = "."
xgb_cv = pickle.load(open(PATH + "/model/xgb_pipeline_model.pkl", "rb"))
x_train_d1 = pickle.load(open(PATH + "/model/x_train_d2", "rb"))
jobs = {}

# Inicializar FastAPI
app = FastAPI(
    title="EEG Model API",
    description="Endpoints: /features (EDF->JSON), \
    /predict (JSON->predicción), /explain (JSON->LIME)"
)

# ======================
# Funciones auxiliares
# ======================
def extraer_canal(datos, nombre_canal):
    """ Extrae los datos de un canal específico de un archivo EDF."""
    canal = pd.DataFrame(datos[datos.ch_names.index(nombre_canal)][0])
    return canal.T.values

def procesar_edf(file_path, fs=250):
    """ Procesa un archivo EDF y extrae características de bandas EEG."""
    raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=None)
    channels = raw.ch_names
    pattern = r'(Fp[12]|F[23478]|C[34]|P[34]|O[12]|T[34]|T[56]|A1|A2|FP[12])' \
              r'-?(A1|A2|C4|P4|O2|Fp[12]|F[3478]|C[34]|P[34]|O[12]|T[34]|T[56])?'
    eeg_channels = [ch for ch in channels if re.match(pattern, ch)]

    bandas_eeg = {
        'Delta': (0.5, 3.9),
        'Theta': (4, 7.9),
        'Alpha': (8, 12.9),
        'Beta': (13, 29.9),
        'Gamma': (30, 80)
    }

    df = pd.DataFrame(columns=["canal", 'delta', 'theta', 'alpha', 'beta', 'gamma'])

    for ch in eeg_channels:
        arr_canal = extraer_canal(raw, ch)
        fft_vals = np.abs(np.fft.rfft(arr_canal))
        fft_freq = np.fft.rfftfreq(len(arr_canal), 1.0/fs)

        features = []
        for band in bandas_eeg:
            freq_ix = np.where(
                (fft_freq >= bandas_eeg[band][0]) &
                (fft_freq <= bandas_eeg[band][1])
            )[0]
            features.append(np.mean(fft_vals[freq_ix]))

        df.loc[len(df)] = [ch] + features

    return df

def predict_fn_wrapped(data):
    """ Función de predicción para LIME, envuelve el modelo XGBoost."""
    data_df = pd.DataFrame(data, columns=x_train_d1.columns)
    return xgb_cv.predict_proba(data_df)

# ======================
# 2. Endpoint de extracción de features
# ======================

# Función que hace el trabajo pesado
def process_features_job(job_id: str, file_path: str):
    """ Procesa un archivo EDF en segundo plano y almacena el resultado."""
    try:
        df = procesar_edf(file_path)
        os.remove(file_path)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = df.to_dict(orient="records")
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/features")
async def create_features_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """ Recibe un archivo EDF y devuelve las características de bandas EEG en JSON."""
    # Guardar temporalmente el archivo
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Crear job_id
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "result": None}

    # Ejecutar en segundo plano
    background_tasks.add_task(process_features_job, job_id, temp_path)

    return {"job_id": job_id, "status": "processing"}

@app.get("/features/{job_id}")
async def get_features_result(job_id: str):
    """ Devuelve el resultado de un job de extracción de características."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]
# ======================
# 3. Endpoint de predicción (recibe JSON de un canal)
# ======================
@app.post("/predict", response_model=PredictionOutput, summary="Clasificación EEG", tags=["Modelo"])
async def predict(features: FeatureInput):
    """
    Predice la clase de un canal EEG a partir de características de bandas.

    - **Delta**, **Theta**, **Alpha**, **Beta**, **Gamma**: valores numéricos obtenidos del análisis FFT.
    """
    input_df = pd.DataFrame([features.dict()], columns=x_train_d1.columns)
    y_pred = xgb_cv.predict(input_df)[0]
    y_proba = xgb_cv.predict_proba(input_df)[0].tolist()

    return {
        "prediction": str(y_pred),
        "probabilities": y_proba
    }

# ======================
# 4. Endpoint de explicación (recibe JSON de un canal)
# ======================
@app.post("/explain", summary="Explicación LIME", tags=["Modelo"])
async def explain(features: FeatureInput, return_html: bool = True):
    """
    Genera una explicación LIME del modelo sobre un canal EEG.

    - Si `return_html=true`, devuelve HTML embebido.
    - Si `false`, devuelve un JSON con pesos por feature.
    """
    input_df = pd.DataFrame([features.dict()], columns=x_train_d1.columns)
    explainer = LimeTabularExplainer(
        x_train_d1.values,
        mode="classification",
        class_names=['n', 'sleep_dis', 'neurological'],
        feature_names=x_train_d1.columns
    )

    explanation = explainer.explain_instance(
        data_row=input_df.iloc[0].values,
        predict_fn=predict_fn_wrapped,
        num_features=input_df.shape[1]
    )

    if return_html:
        file_id = str(uuid.uuid4())
        out_file = f"./lime_{file_id}.html"
        explanation.save_to_file(out_file)

        with open(out_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        os.remove(out_file)
        return HTMLResponse(content=html_content)
    else:
        exp_map = explanation.as_map()
        feature_names = x_train_d1.columns
        exp_dict = {feature_names[int(f)]: float(w) for f, w in exp_map[1]}
        return JSONResponse(exp_dict)
