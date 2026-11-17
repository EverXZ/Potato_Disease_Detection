# Potato Disease Detection (AgroScan AI)

Sistema web inteligente para la **detección de enfermedades en hojas de papa** (*Solanum tuberosum*) que combina visión por computadora (CNN), generación de recomendaciones agronómicas mediante LLM y adaptación continua con Aprendizaje por Refuerzo (RL).

---

## ¿Qué hace el proyecto?

1. **Diagnóstico visual de hojas:** El usuario sube una imagen de una hoja de papa y una red neuronal convolucional (`.keras`) clasifica su estado en una de tres clases:
   - **Tizón Temprano** (`Potato___Early_blight` — *Alternaria solani*)
   - **Tizón Tardío** (`Potato___Late_blight` — *Phytophthora infestans*)
   - **Planta Sana** (`Potato___healthy`)
2. **Recomendación agronómica personalizada:** Un agente de Aprendizaje por Refuerzo elige la estrategia de comunicación más efectiva (`tecnico`, `practico` o `ecologico`) y una API de LLM (Google Gemini o DeepSeek, con respaldo local automático) redacta la guía de acción y habilita un chat de seguimiento.
3. **Aprendizaje continuo por feedback:** El usuario califica la recomendación de 1 a 5 estrellas y el agente actualiza su tabla `Q(s, a)` para mejorar futuras decisiones.

---

## ¿Qué tiene por dentro?

La aplicación principal vive en `modest-bell/` y está organizada en módulos desacoplados:

- **`modest-bell/app.py`** — Servidor **FastAPI** que expone la interfaz web y los endpoints `POST /predict`, `POST /feedback`, `POST /regenerate`, `POST /chat`, `GET /rl-stats` y `GET /model-status`. Carga el modelo `.keras` (o activa un modo demostración si no hay archivo de pesos).
- **`modest-bell/rl_agent.py`** — Agente de **Aprendizaje por Refuerzo (Contextual Multi-Armed Bandit)** con política **ε-greedy** ($\varepsilon = 0.2$, $\alpha = 0.2$). Normaliza calificaciones de 1–5 estrellas a recompensas $r \in [-1, +1]$ y actualiza los valores `Q(s, a)` con escritura atómica.
- **`modest-bell/llm_service.py`** — Orquestador de llamadas a **Google Gemini** (`google-genai`) y **DeepSeek** (`openai` SDK). Construye los prompts según la estrategia elegida por el agente RL y cuenta con un sistema de contingencia (`build_structured_fallback`) si falla la red o falta la API key.
- **`modest-bell/data/rl_policy.json`** — Persistencia en JSON de la tabla `Q(s, a)`, conteos por estado/acción, métricas globales e historial reciente de feedback.
- **`modest-bell/templates/index.html` y `modest-bell/static/`** — Interfaz web construida con HTML5, Tailwind CSS y JavaScript asíncrono (`fetch`), incluyendo carga de imágenes, barras de probabilidad, widget de calificación y chat de seguimiento.
- **`modest-bell/explicacion-arquitectura-rl-api.html`** — Explicación visual interactiva del funcionamiento interno de la API, el agente RL y el flujo de feedback.

---

## Inicio rápido

```bash
cd modest-bell
pip install -r requirements.txt
python app.py
```

Luego abre `http://localhost:8000` en tu navegador. Para más detalles de ejecución, consulta [modest-bell/README.md](modest-bell/README.md).
