
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_flores.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"

IMG_SIZE = (128, 128)

st.set_page_config(
    page_title="Clasificador de flores",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Clasificador de flores")
st.write(
    "Carga una imagen y el modelo de Deep Learning "
    "identificará el tipo de flor."
)

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Imagen cargada", width="stretch")

    if st.button("Clasificar flor", type="primary"):
        imagen_redimensionada = imagen.resize(IMG_SIZE)

        array = np.array(
            imagen_redimensionada,
            dtype=np.float32
        ) / 255.0

        array = np.expand_dims(array, axis=0)

        predicciones = model.predict(array, verbose=0)[0]

        indice = int(np.argmax(predicciones))
        confianza = float(predicciones[indice])

        nombres_es = {
            "dandelion": "Diente de león",
            "daisy": "Margarita",
            "tulips": "Tulipanes",
            "sunflowers": "Girasoles",
            "roses": "Rosas"
        }

        clase = class_names[indice]

        st.success(
            f"Predicción: {nombres_es.get(clase, clase)}"
        )
        st.metric(
            "Nivel de confianza",
            f"{confianza:.2%}"
        )

        st.subheader("Probabilidades por clase")

        for i, nombre in enumerate(class_names):
            st.write(nombres_es.get(nombre, nombre))
            st.progress(float(predicciones[i]))
            st.caption(f"{predicciones[i]:.2%}")
