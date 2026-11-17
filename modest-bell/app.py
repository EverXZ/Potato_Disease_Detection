import os
import io
import json
from pathlib import Path
from typing import Optional

# Configurar backend de Keras antes de importar keras
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Intentar importar keras de forma segura
try:
    import keras
    KERAS_AVAILABLE = True
except Exception as e:
    keras = None
    KERAS_AVAILABLE = False
    print(f"[AVISO] Keras no está disponible directamente: {e}")

app = FastAPI(
    title="Detector de Enfermedades de la Papa",
    description="Aplicación web para diagnosticar patologías en hojas de papa mediante un modelo .keras",
    version="1.0.0"
)

# Configurar rutas de carpetas
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Clases del dataset PlantVillage
CLASS_INFO = {
    0: {
        "id": "Potato___Early_blight",
        "name": "Tizón Temprano",
        "scientific_name": "Alternaria solani",
        "severity": "Moderada / Alta",
        "color": "amber",
        "description": "Enfermedad fúngica caracterizada por manchas foliares circulares u ovales con anillos concéntricos ('diana' u 'ojo de buey'), rodeadas a menudo de un halo clorótico amarillo.",
        "recommendations": [
            "Aplicar fungicidas preventivos/curativos como Mancozeb, Clorotalonil o a base de Cobre.",
            "Evitar el riego por aspersión que moje las hojas prolongadamente.",
            "Eliminar y destruir restos vegetales infectados para reducir inóculo.",
            "Rotar cultivos con especies no solanáceas por al menos 2 años."
        ]
    },
    1: {
        "id": "Potato___Late_blight",
        "name": "Tizón Tardío",
        "scientific_name": "Phytophthora infestans",
        "severity": "Crítica / Muy Alta",
        "color": "red",
        "description": "Oomiceto altamente agresivo y destructivo. Produce lesiones necróticas de color verde oscuro a marrón negruzco de aspecto acuoso. Bajo humedad alta genera un fino vello blanco en el envés.",
        "recommendations": [
            "¡Acción inmediata! Aplicar fungicidas sistémicos específicos (Metalaxil, Cimoxanilo o Dimetomorf).",
            "Destruir focos iniciales de infección para evitar propagación por esporangios en el viento.",
            "Mantener buena aireación entre surcos y monitorear el follaje en temporadas frías y húmedas.",
            "Utilizar variedades de semilla certificada y tolerantes a la enfermedad."
        ]
    },
    2: {
        "id": "Potato___healthy",
        "name": "Planta Sana",
        "scientific_name": "Solanum tuberosum (Saludable)",
        "severity": "Ninguna",
        "color": "emerald",
        "description": "La hoja no presenta síntomas visibles de lesiones fúngicas, clorosis irregular ni tejido necrótico. Pigmentación verde homogénea.",
        "recommendations": [
            "Mantener el plan de fertilización equilibrado (evitar exceso de nitrógeno).",
            "Continuar con el monitoreo preventivo semanal del cultivo.",
            "Asegurar un drenaje adecuado del suelo y manejo integrado de plagas vectores."
        ]
    }
}

# Variable global para el modelo cargado
model = None
model_source_path = None

def find_model_file() -> Optional[Path]:
    """Busca un archivo .keras en la carpeta models/ o en el directorio raíz."""
    # 1. Buscar en models/
    keras_files = list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5"))
    if keras_files:
        return keras_files[0]
    
    # 2. Buscar en la raíz del proyecto
    root_files = list(BASE_DIR.glob("*.keras")) + list(BASE_DIR.glob("*.h5"))
    if root_files:
        return root_files[0]
        
    return None

def load_potato_model():
    """Carga el modelo Keras si está disponible."""
    global model, model_source_path
    
    if not KERAS_AVAILABLE:
        print("[INFO] Keras no está disponible en este entorno. Corriendo en modo demostración/simulación.")
        return None
        
    model_path = find_model_file()
    if model_path and model_path.exists():
        try:
            print(f"[INFO] Cargando modelo Keras desde: {model_path} ...")
            # Usar keras.saving.load_model de Keras 3
            model = keras.saving.load_model(str(model_path), compile=False)
            model_source_path = str(model_path.name)
            print("[INFO] ¡Modelo Keras cargado exitosamente!")
            return model
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el archivo {model_path}: {e}")
            model = None
            model_source_path = None
    else:
        print("[INFO] No se encontró ningún archivo .keras en la carpeta 'models/'. Modo demostración activo.")
        model = None
        model_source_path = None
        
    return model

# Intentar cargar al iniciar
load_potato_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renderiza la página web principal."""
    has_model = (model is not None)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "has_model": has_model,
            "model_name": model_source_path or "No cargado (Modo Demo Activo)",
            "classes": CLASS_INFO
        }
    )

@app.get("/favicon.ico")
async def favicon():
    """Sirve el favicon en formato SVG."""
    favicon_path = STATIC_DIR / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"error": "Favicon not found"})

@app.get("/model-status")
async def get_model_status():
    """Devuelve el estado actual del modelo cargado."""
    return {
        "has_model": model is not None,
        "model_file": model_source_path,
        "keras_available": KERAS_AVAILABLE
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen de hoja de papa, la preprocesa a 224x224
    y devuelve la predicción con probabilidades y recomendaciones.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "El archivo proporcionado no es una imagen válida."}
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        global model
        if model is None:
            # Reintentar cargar por si el usuario acaba de pegar el archivo .keras
            load_potato_model()

        # Detectar tamaño de entrada requerido por el modelo (ej. 256x256 o 224x224)
        target_size = (256, 256)
        if model is not None and hasattr(model, "input_shape") and model.input_shape:
            shape = model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                target_size = (int(shape[1]), int(shape[2]))

        image_resized = image.resize(target_size)
        img_array = np.array(image_resized, dtype=np.float32)
        
        # Expandir dimensiones a lote (batch_size=1, H, W, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        if model is not None:
            # Inferencia real con el modelo Keras
            try:
                import torch
                with torch.no_grad():
                    predictions = model(img_batch, training=False)
            except Exception:
                predictions = model(img_batch, training=False)

            # Extraer a numpy de forma segura desprendiendo el grafo de gradientes
            if hasattr(predictions, "detach"):
                predictions = predictions.detach().cpu().numpy()
            elif hasattr(predictions, "numpy"):
                predictions = predictions.numpy()
            else:
                predictions = np.array(predictions)

            probs = np.array(predictions[0], dtype=np.float32)
            # Si no suma 1 (e.g. logits), aplicar softmax
            if np.abs(np.sum(probs) - 1.0) > 0.05:
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()
            is_demo = False
        else:
            # Modo demostración cuando aún no se ha colocado el archivo .keras
            # Genera probabilidades coherentes para probar la interfaz visual
            # Analiza color medio de la imagen para dar una simulación verosímil
            r, g, b = np.mean(img_array[:, :, 0]), np.mean(img_array[:, :, 1]), np.mean(img_array[:, :, 2])
            if g > r * 1.2:
                # Muy verde -> Muy probable planta sana
                probs = np.array([0.05, 0.03, 0.92], dtype=np.float32)
            elif r > g:
                # Tonos marrones/oscuros -> Tizón tardío
                probs = np.array([0.15, 0.81, 0.04], dtype=np.float32)
            else:
                # Tizón temprano
                probs = np.array([0.88, 0.09, 0.03], dtype=np.float32)
            is_demo = True

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        info = CLASS_INFO[predicted_idx]

        # Formatear desglose de probabilidades para todas las clases (5 decimales)
        breakdown = []
        for idx, p in enumerate(probs):
            breakdown.append({
                "class_idx": idx,
                "name": CLASS_INFO[idx]["name"],
                "scientific_name": CLASS_INFO[idx]["scientific_name"],
                "probability": float(round(p * 100, 5)),
                "color": CLASS_INFO[idx]["color"]
            })
        
        # Ordenar de mayor a menor probabilidad
        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "success": True,
            "is_demo": is_demo,
            "model_file": model_source_path or "Modo Demostración (Coloca tu .keras en models/)",
            "prediction": {
                "class_index": predicted_idx,
                "disease_name": info["name"],
                "scientific_name": info["scientific_name"],
                "severity": info["severity"],
                "color": info["color"],
                "confidence": round(confidence * 100, 5),
                "description": info["description"],
                "recommendations": info["recommendations"]
            },
            "breakdown": breakdown
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al procesar la imagen: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
