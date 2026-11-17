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
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

import llm_service
from rl_agent import ACTIONS, ContextualBanditAgent

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

# Configurar rutas de carpetas y variables de entorno
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False, encoding="utf-8")

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Agente de Aprendizaje por Refuerzo (Contextual Multi-Armed Bandit)
rl_agent = ContextualBanditAgent(policy_path=DATA_DIR / "rl_policy.json")


class FeedbackRequest(BaseModel):
    disease_id: str = Field(..., min_length=1, max_length=128)
    strategy: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(default="", max_length=500)

    @field_validator("rating", mode="before")
    @classmethod
    def reject_bool_rating(cls, value):
        if isinstance(value, bool):
            raise ValueError("La calificación debe ser un número entero entre 1 y 5, no booleano.")
        return value


class RegenerateRequest(BaseModel):
    disease_id: str = Field(..., min_length=1, max_length=128)
    disease_name: str = Field(..., min_length=1, max_length=128)
    scientific_name: str = Field(..., min_length=1, max_length=128)
    confidence: float = Field(default=90.0)
    severity: str = Field(default="Moderada")
    breakdown: list[dict] = Field(default_factory=list)
    strategy: str = Field(default="auto")


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    disease_id: str = Field(..., min_length=1, max_length=128)
    disease_name: str = Field(..., min_length=1, max_length=128)
    scientific_name: str = Field(..., min_length=1, max_length=128)
    confidence: float = Field(default=90.0)
    severity: str = Field(default="Moderada")
    strategy: str = Field(default="practico")
    initial_recommendation: str = Field(default="")
    history: list[ChatTurn] = Field(default_factory=list)
    message: str = Field(..., min_length=1, max_length=1500)

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
async def predict(
    file: UploadFile = File(...),
    strategy_override: Optional[str] = Form(default="auto"),
):
    """
    Recibe una imagen de hoja de papa, la preprocesa a 224x224
    y devuelve la predicción con probabilidades y recomendaciones.
    """
    if not (file.content_type or "").startswith("image/"):
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
                "disease_id": CLASS_INFO[idx]["id"],
                "name": CLASS_INFO[idx]["name"],
                "scientific_name": CLASS_INFO[idx]["scientific_name"],
                "probability": float(round(p * 100, 5)),
                "color": CLASS_INFO[idx]["color"]
            })
        
        # Ordenar de mayor a menor probabilidad
        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        confidence_pct = round(confidence * 100, 5)

        # Selección de estrategia mediante agente RL o selección manual del usuario
        clean_override = (strategy_override or "auto").strip().lower()
        rl_info = rl_agent.select_action(info["id"])
        if clean_override in ACTIONS:
            rl_info["strategy"] = clean_override
            rl_info["mode"] = "manual"
            rl_info["mode_label"] = "Selección Manual"
            rl_info["is_exploration"] = False

        llm_result = llm_service.generate_recommendation(
            disease_id=info["id"],
            disease_name=info["name"],
            scientific_name=info["scientific_name"],
            confidence=confidence_pct,
            severity=info["severity"],
            breakdown=breakdown,
            strategy=rl_info["strategy"],
        )

        return {
            "success": True,
            "is_demo": is_demo,
            "model_file": model_source_path or "Modo Demostración (Coloca tu .keras en models/)",
            "prediction": {
                "class_index": predicted_idx,
                "disease_id": info["id"],
                "disease_name": info["name"],
                "scientific_name": info["scientific_name"],
                "severity": info["severity"],
                "color": info["color"],
                "confidence": confidence_pct,
                "description": info["description"],
                "recommendations": info["recommendations"]
            },
            "breakdown": breakdown,
            "llm_recommendation": llm_result["recommendation"],
            "llm_info": {
                "provider": llm_result["provider"],
                "model": llm_result["model"],
                "strategy": llm_result["strategy"],
                "char_count": llm_result["char_count"],
                "is_fallback": llm_result["is_fallback"],
                "fallback_reason": llm_result["fallback_reason"],
            },
            "rl_info": rl_info,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al procesar la imagen: {str(e)}"}
        )


@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    """Registra la calificación de 1 a 5 estrellas y actualiza la tabla Q(s, a) del agente RL."""
    disease_id = payload.disease_id.strip()
    if not disease_id:
        return JSONResponse(
            status_code=400,
            content={"error": "El campo 'disease_id' no puede estar vacío."}
        )
    if payload.strategy not in ACTIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Estrategia inválida: {payload.strategy}. Valores permitidos: {', '.join(ACTIONS)}."}
        )

    try:
        update_result = rl_agent.update(
            state=disease_id,
            action=payload.strategy,
            rating=payload.rating,
            comment=payload.comment,
        )
        return {
            "success": True,
            "update": update_result,
            "stats": rl_agent.get_stats(),
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al registrar retroalimentación: {str(e)}"}
        )


@app.get("/rl-stats")
async def get_rl_stats():
    """Devuelve las métricas en vivo, tabla Q(s, a), conteos y feedback reciente del agente RL."""
    return rl_agent.get_stats()


@app.post("/regenerate")
async def regenerate_recommendation(payload: RegenerateRequest):
    """Permite cambiar el modo de estrategia (o pedirle al agente RL otra selección) sin volver a subir la imagen."""
    clean_strategy = (payload.strategy or "auto").strip().lower()
    rl_info = rl_agent.select_action(payload.disease_id.strip())
    if clean_strategy in ACTIONS:
        rl_info["strategy"] = clean_strategy
        rl_info["mode"] = "manual"
        rl_info["mode_label"] = "Selección Manual"
        rl_info["is_exploration"] = False

    llm_result = llm_service.generate_recommendation(
        disease_id=payload.disease_id.strip(),
        disease_name=payload.disease_name.strip(),
        scientific_name=payload.scientific_name.strip(),
        confidence=payload.confidence,
        severity=payload.severity.strip(),
        breakdown=payload.breakdown,
        strategy=rl_info["strategy"],
    )
    return {
        "success": True,
        "llm_recommendation": llm_result["recommendation"],
        "llm_info": {
            "provider": llm_result["provider"],
            "model": llm_result["model"],
            "strategy": llm_result["strategy"],
            "char_count": llm_result["char_count"],
            "is_fallback": llm_result["is_fallback"],
            "fallback_reason": llm_result["fallback_reason"],
        },
        "rl_info": rl_info,
    }


@app.post("/chat")
async def chat_with_assistant(payload: ChatRequest):
    """Continúa la conversación con el asistente IA sobre el diagnóstico actual."""
    question = payload.message.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "El mensaje no puede estar vacío."})

    chat_result = llm_service.chat_followup(
        disease_id=payload.disease_id.strip(),
        disease_name=payload.disease_name.strip(),
        scientific_name=payload.scientific_name.strip(),
        confidence=payload.confidence,
        severity=payload.severity.strip(),
        strategy=payload.strategy.strip(),
        initial_recommendation=payload.initial_recommendation,
        history=[{"role": t.role, "content": t.content} for t in payload.history],
        user_message=question,
    )
    return {
        "success": True,
        "reply": chat_result["reply"],
        "llm_info": {
            "provider": chat_result["provider"],
            "model": chat_result["model"],
            "strategy": chat_result["strategy"],
            "char_count": chat_result["char_count"],
            "is_fallback": chat_result["is_fallback"],
            "fallback_reason": chat_result["fallback_reason"],
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
