""" Procesa archivos EDF y extrae características de bandas EEG."""
import pickle
import re
import random
import pandas as pd
import numpy as np
import mne
from lime.lime_tabular import LimeTabularExplainer
import imblearn
import xgboost as xgb

# ======================
# 1. Cargar modelo
# ======================

PATH = r"."
xgb_cv = pickle.load(open(PATH+r"/model/xgb_pipeline_model.pkl", "rb"))
x_train_d1 = pickle.load(open(PATH+r"/model/x_train_d2", 'rb'))

# ======================
# 2. Función para extraer features FFT de un EDF
# ======================

def extraer_canal(datos, nombre_canal):
    """ Extrae los datos de un canal específico de un archivo EDF."""
    canal = pd.DataFrame(datos[
        datos.ch_names.index(nombre_canal)
    ][0])
    valores_canal = canal.T
    valores_canal.describe()
    datos = valores_canal.values

    return datos

def procesar_edf(file_path, filename, fs=250):
    """ Procesa un archivo EDF y extrae características de bandas EEG."""
    raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=None)
    channels = raw.ch_names
    pattern = r'(Fp[12]|F[23478]|C[34]|P[34]|O[12]|T[34]|T[56]|A1|A2|FP[12])'+\
        r'-?(A1|A2|C4|P4|O2|Fp[12]|F[3478]|C[34]|P[34]|O[12]|T[34]|T[56])?'
    eeg_channels = [ch for ch in channels if re.match(pattern, ch)]

    bandas_eeg = {
        'delta': (0.5, 3.9),
        'theta': (4, 7.9),
        'alpha': (8, 12.9),
        'beta': (13, 29.9),
        'gamma': (30, 80),
    }

    arr_canal = []
    columns = [
        "canal",'delta', 'theta', 'alpha', 'beta', 'gamma'
    ]
    df = pd.DataFrame(columns=columns)

    for ch in eeg_channels:
        arr_canal = extraer_canal(raw, ch)
        fft_vals = np.absolute(np.fft.rfft(arr_canal))
        fft_freq = np.fft.rfftfreq(len(arr_canal), 1.0/fs)

        bandas_eeg = {'Delta': (0.5, 3.9),
                 'Theta': (4, 7.9),
                 'Alpha': (8, 12.9),
                 'Beta': (13, 29.9),
                 'Gamma': (30, 80)}
        # list_features = []
        features = []
        for band in bandas_eeg:
            freq_ix = np.where(
                (fft_freq >= bandas_eeg[band][0]) &
                (fft_freq <= bandas_eeg[band][1])
            )[0]
            features += [np.mean(fft_vals[freq_ix])]
        # list_features.append(features)
        # print(list_features[0])
        # print(features)
        df.loc[len(df)] = [ch] + features

    return df

# ======================
# 2.5. Extraccion de caracteristicas
# ======================
EDF_PATH = r"./edf/sdb4.edf"
features = procesar_edf(EDF_PATH, "sdb4.edf")
# ======================
# 3. Predicción + Explicación LIME
# ======================
print("Features generados:", features.shape)
print(features.head())

# Predicciones canal por canal
# y_pred = xgb_cv.predict(features)
# y_proba = xgb_cv.predict_proba(features)

# print("Predicciones:", y_pred)

class_names = ['n', 'sleep_dis', 'neurological']
explainer = LimeTabularExplainer(
    x_train_d1.values,
    mode="classification",
    class_names=class_names,
    feature_names=x_train_d1.columns
)

# Elegir un canal aleatorio para explicar
idx = random.randint(0, len(features)-1)
print(f"Explicando canal #{idx}")

# Create a DataFrame with the correct shape and column names
PRED_FEATURES = features.drop(columns=["canal"])
input_data = pd.DataFrame(
    PRED_FEATURES.values[idx].reshape(1, -1), columns=PRED_FEATURES.columns
)

# Use the best estimator to make a prediction
print("Prediction : ", xgb_cv.predict(input_data))
#print("Actual :     ", features[idx])

def predict_fn_wrapped(data):
    """Convert the data to a DataFrame with the correct column names"""
    data_df = pd.DataFrame(data, columns=x_train_d1.columns)
    predictions = xgb_cv.predict_proba(data_df)
    return predictions


explanation = explainer.explain_instance(
    data_row=PRED_FEATURES.iloc[idx].values,
    predict_fn=predict_fn_wrapped,
    num_features=PRED_FEATURES.shape[1]
)
# explanation.show_in_notebook()
# O guardar como HTML
explanation.save_to_file(r"./result/lime_explanation.html")
