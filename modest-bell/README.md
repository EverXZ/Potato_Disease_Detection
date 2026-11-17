# 🥔 AgroScan: Detector de Enfermedades de la Papa (.keras)

Aplicación web profesional para diagnosticar patologías en hojas de papa utilizando un modelo de Deep Learning entrenado (`.keras`), construida con **FastAPI**, **HTML5/Tailwind CSS** y **JavaScript asíncrono** (sin usar Gradio).

---

## 🚀 Cómo Iniciar la Aplicación

### 1. Requisitos
Asegúrate de tener instaladas las dependencias:
```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Servidor Web
En la terminal, dentro de esta carpeta, ejecuta:
```bash
python -m uvicorn app:app --reload --port 8000
```
O directamente:
```bash
python app.py
```

Abre tu navegador en:  
👉 **`http://localhost:8000`**

---

## 📁 Dónde colocar tu modelo entrenado

1. En tu notebook de Google Colab, tras entrenar tu red neuronal, descárgala con:
   ```python
   from google.colab import files
   modelo_papa_v3.save('mejor_modelo_papa_cnn.keras')
   files.download('mejor_modelo_papa_cnn.keras')
   ```
2. Coloca el archivo `.keras` dentro de la carpeta:  
   📂 **`models/`**
3. ¡Listo! La aplicación detecta el modelo automáticamente y realiza inferencias con tus pesos reales. Si aún no has pegado el archivo, la aplicación funciona en **Modo Demostración interactivo** para que pruebes toda la interfaz gráfica.

---

## 🌐 Alternativas para ejecutar modelos en la Web (Sin Gradio)

| Alternativa | Tipo de Arquitectura | Ventajas | Desventajas | Cuándo usarla |
| :--- | :--- | :--- | :--- | :--- |
| **FastAPI + HTML/Tailwind/JS** *(Implementada)* | Full-Stack (Backend API + Frontend) | • Diseño 100% personalizable.<br>• Separación limpia frontend/backend.<br>• Asíncrono, ultrarrápido y estándar de la industria. | Requiere un mínimo conocimiento de HTML/JS si quieres editar el diseño. | **Ideal para proyectos universitarios serios, startups y aplicaciones de producción.** |
| **Streamlit** *(Incluida en `streamlit_app.py`)* | Python UI pura | • 100% en Python con 5 líneas de código.<br>• Muy fácil para Data Scientists.<br>• Componentes prefabricados. | Menor control sobre el diseño visual y animaciones avanzadas. | **Prototipado rápido y dashboards interactivos.** |
| **Flask + Bootstrap** | Full-Stack tradicional | • Muy conocido en la comunidad.<br>• Curva de aprendizaje baja. | Síncrono por defecto; más código repetitivo que FastAPI. | Si tu equipo ya domina Flask y prefiere no usar frameworks asíncronos. |
| **TensorFlow.js (TFJS)** | 100% Navegador (Client-side) | • **Cero costo de servidor:** la red corre en la CPU/GPU del usuario.<br>• Se puede alojar gratis en GitHub Pages o Vercel.<br>• Privacidad total (la imagen no sale del dispositivo). | La primera carga debe descargar los pesos del modelo en el navegador. | **Si quieres publicar tu web gratis en GitHub Pages sin pagar servidor.** |
| **Docker + Cloud Run / Render / HF Spaces** | Despliegue en la Nube | • Obtienes un enlace HTTPS público para compartir con profesores o clientes.<br>• Escalable y reproducible. | Requiere configurar una cuenta en la plataforma cloud. | **Para presentar la entrega final en vivo con un enlace accesible desde cualquier celular.** |
