import os
import io
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st

# Configurar backend si es necesario
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

try:
    import keras
    KERAS_AVAILABLE = True
except Exception:
    keras = None
    KERAS_AVAILABLE = False

st.set_page_config(
    page_title="AgroScan - Diagnóstico de Hojas de Papa",
    page_icon="🌱",
    layout="centered"
)

# Diccionario de clases
CLASS_INFO = {
    0: {
        "name": "Tizón Temprano",
        "scientific": "Alternaria solani",
        "desc": "Hongo que causa manchas circulares concéntricas en las hojas ('diana').",
        "action": "Aplicar Mancozeb o fungicidas cúpricos. Evitar mojar follaje en riego."
    },
    1: {
        "name": "Tizón Tardío",
        "scientific": "Phytophthora infestans",
        "desc": "Oomiceto altamente destructivo con manchas acuosas y necrosis rápida.",
        "action": "¡Acción urgente! Aplicar fungicidas sistémicos (Metalaxil/Cimoxanilo) y eliminar focos."
    },
    2: {
        "name": "Planta Sana",
        "scientific": "Solanum tuberosum",
        "desc": "Follaje sin signos de patógenos ni necrosis foliar.",
        "action": "Mantener monitoreo regular y plan de fertilización equilibrado."
    }
}

@st.cache_resource
def load_model():
    models_dir = Path("models")
    keras_files = list(models_dir.glob("*.keras")) + list(Path(".").glob("*.keras"))
    if keras_files and KERAS_AVAILABLE:
        try:
            return keras.saving.load_model(str(keras_files[0]), compile=False), keras_files[0].name
        except Exception as e:
            st.warning(f"Error al cargar el modelo: {e}")
    return None, None

model, model_name = load_model()

st.title("🥔 AgroScan: Diagnóstico en Hojas de Papa")
st.caption("Aplicación en Streamlit para inferencia con modelos .keras")

if model_name:
    st.success(f"Modelo cargado: `{model_name}`")
else:
    st.info("ℹ️ Modo demostración activo. Coloca tu archivo `.keras` en la carpeta `models/`.")

uploaded_file = st.file_uploader(
    "Sube una fotografía de una hoja de papa",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Diagnosticar Hoja", type="primary"):
        with st.spinner("Analizando con el modelo..."):
            target_size = (256, 256)
            if model is not None and hasattr(model, "input_shape") and model.input_shape:
                shape = model.input_shape
                if isinstance(shape, list):
                    shape = shape[0]
                if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                    target_size = (int(shape[1]), int(shape[2]))

            img_resized = image.resize(target_size)
            img_array = np.array(img_resized, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)

            if model is not None:
                try:
                    import torch
                    with torch.no_grad():
                        preds = model(img_batch, training=False)
                except Exception:
                    preds = model(img_batch, training=False)

                if hasattr(preds, "detach"):
                    preds = preds.detach().cpu().numpy()
                elif hasattr(preds, "numpy"):
                    preds = preds.numpy()
                else:
                    preds = np.array(preds)
                probs = np.array(preds[0], dtype=np.float32)
            else:
                # Simulación coherente si no hay modelo físico
                r, g, b = np.mean(img_array[:, :, 0]), np.mean(img_array[:, :, 1]), np.mean(img_array[:, :, 2])
                if g > r * 1.2:
                    probs = np.array([0.05, 0.03, 0.92])
                elif r > g:
                    probs = np.array([0.15, 0.81, 0.04])
                else:
                    probs = np.array([0.88, 0.09, 0.03])

            best_idx = int(np.argmax(probs))
            best_info = CLASS_INFO[best_idx]
            conf = probs[best_idx] * 100

            st.divider()
            if best_idx == 2:
                st.success(f"### Resultado: {best_info['name']} ({conf:.5f}%)")
            elif best_idx == 0:
                st.warning(f"### Resultado: {best_info['name']} ({conf:.5f}%)")
            else:
                st.error(f"### Resultado: {best_info['name']} ({conf:.5f}%)")

            st.write(f"**Patógeno:** *{best_info['scientific']}*")
            st.write(f"**Descripción:** {best_info['desc']}")
            st.info(f"💡 **Recomendación:** {best_info['action']}")

            st.write("#### Probabilidades detalladas:")
            for i in range(3):
                p = probs[i] * 100
                st.write(f"- **{CLASS_INFO[i]['name']}**: `{p:.5f}%`")
                st.progress(min(float(probs[i]), 1.0))
