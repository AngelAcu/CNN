# Clasificación de flores con Deep Learning

## Descripción

Proyecto de clasificación de imágenes mediante una Red Neuronal Convolucional (CNN). Se utiliza el dataset TF Flowers para reconocer cinco categorías de flores: **diente de león, margarita, tulipán, girasol y rosa.**

El proyecto compara una CNN construida desde cero con MobileNetV2 mediante transferencia de aprendizaje.

## Dataset

- Fuente: TensorFlow Datasets, `tf_flowers`.
- Total: 3.670 imágenes.
- Entrenamiento: 2.569 imágenes.
- Validación: 551 imágenes.
- Prueba: 550 imágenes.
- Tamaño de entrada: 128 × 128 píxeles.

## Preprocesamiento

Las imágenes se redimensionan a 128 × 128 y se normalizan. Durante el entrenamiento se utiliza aumento de datos mediante volteo horizontal, rotación y zoom.

## Resultados

El modelo MobileNetV2 obtuvo una precisión de **87,64 % en el conjunto de prueba**.

Las métricas adicionales, la matriz de confusión y las predicciones de ejemplo se encuentran en la carpeta `results/`.

## Estructura

- `notebooks/`: notebook de experimentación.
- `app/`: aplicación Streamlit.
- `models/`: modelo entrenado y nombres de las clases.
- `results/`: gráficas, matriz de confusión y predicciones.
- `requirements.txt`: dependencias del proyecto.

## Paso a paso para ejecutar el proyecto

Ejecuta los siguientes comandos en orden desde la carpeta raíz del proyecto clasificacion-flores.

1. Crear el entorno virtual

python -m venv .venv

2. Activar el entorno virtual

.\.venv\Scripts\Activate.ps1

3. Instalar las dependencias

python -m pip install -r requirements.txt

4. Ejecutar la aplicación Streamlit

python -m streamlit run app/streamlit_app.py

## Uso

1. Ejecuta la aplicación.
2. Carga una imagen en formato JPG, JPEG o PNG.
3. Presiona el botón para clasificarla.
4. Consulta la clase predicha y las probabilidades por categoría.

El modelo reconoce las cinco clases del dataset. Una imagen de otra especie podría clasificarse incorrectamente.