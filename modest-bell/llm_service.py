import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False, encoding="utf-8")

VALID_STRATEGIES: tuple[str, ...] = ("tecnico", "practico", "ecologico")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
)
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
MIN_RECOMMENDATION_CHARS = 200


def _normalize_confidence_pct(confidence: float) -> float:
    try:
        val = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < val <= 1.0:
        val *= 100.0
    return max(0.0, min(100.0, val))


def _normalize_strategy(strategy: Optional[str]) -> str:
    if isinstance(strategy, str):
        cleaned = strategy.strip().lower()
        if cleaned in VALID_STRATEGIES:
            return cleaned
    return "practico"


def _is_valid_api_key(api_key: Optional[str]) -> bool:
    if not api_key or not isinstance(api_key, str):
        return False
    cleaned = api_key.strip()
    if len(cleaned) < 8:
        return False
    lowered = cleaned.lower()
    placeholder_prefixes = (
        "your_",
        "tu_",
        "replace_",
        "example",
        "changeme",
        "placeholder",
        "xxx",
        "<",
    )
    if lowered in {"none", "null", "empty", "api_key"} or lowered.startswith(placeholder_prefixes):
        return False
    return True


def _format_breakdown_summary(breakdown: list[dict[str, Any]]) -> str:
    if not breakdown:
        return "Sin desglose adicional disponible."
    parsed: list[tuple[str, float]] = []
    for item in breakdown[:3]:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("disease_name") or item.get("disease_id") or "Clase"
        prob = item.get("probability", 0.0)
        try:
            prob_val = float(prob)
        except (TypeError, ValueError):
            prob_val = 0.0
        parsed.append((str(name), prob_val))

    if not parsed:
        return "Sin desglose adicional disponible."

    probs = [val for _, val in parsed]
    is_fraction_scale = (
        max(probs) > 0.0
        and all(0.0 <= val <= 1.0 for val in probs)
        and sum(probs) <= 1.05
    )
    scale = 100.0 if is_fraction_scale else 1.0

    parts = [f"{name}: {prob_val * scale:.2f}%" for name, prob_val in parsed]
    return " | ".join(parts)


def build_prompts(
    disease_id: str,
    disease_name: str,
    scientific_name: str,
    confidence_pct: float,
    severity: str,
    breakdown: list[dict[str, Any]],
    strategy: str,
) -> tuple[str, str]:
    """Construye el system_prompt y user_prompt en español según la estrategia RL."""
    clean_strategy = _normalize_strategy(strategy)
    breakdown_text = _format_breakdown_summary(breakdown)

    strategy_instructions = {
        "tecnico": (
            "Adopta el rol de un ingeniero agrónomo fitopatólogo especialista en cultivo de papa (Solanum tuberosum). "
            "Entrega un protocolo técnico preciso en español: menciona ingredientes activos recomendados, rotación de "
            "grupos FRAC para prevenir resistencia, criterios de dosificación e intervalos de aplicación, manejo de "
            "humedad relativa y bioseguridad del lote. Si la planta está sana, detalla un plan técnico de monitoreo "
            "preventivo, balance nutricional NPK y vigilancia epidemiológica."
        ),
        "practico": (
            "Adopta el rol de un asesor agrícola cercano que explica de forma práctica, clara y accionable para un "
            "agricultor en campo. Evita tecnicismos excesivos y organiza la respuesta en pasos concretos que el "
            "productor pueda ejecutar hoy mismo: inspección del follaje, retiro de hojas dañadas, horarios de riego, "
            "limpieza de herramientas y cuidados inmediatos del cultivo."
        ),
        "ecologico": (
            "Adopta el rol de un especialista en agroecología y manejo integrado de plagas (MIP) sostenible. Prioriza "
            "prácticas orgánicas y biológicas en español: podas sanitarias, aireación entre surcos, uso de agentes "
            "biológicos (como Trichoderma spp. o Bacillus subtilis), biopreparados o caldo bordelés dentro de límites "
            "orgánicos cuando aplique, extractos vegetales, rotación de cultivos y salud microbiológica del suelo."
        ),
    }

    system_prompt = (
        "Eres AgroScan AI, un asistente agronómico experto en sanidad del cultivo de papa. "
        f"{strategy_instructions[clean_strategy]} "
        "Responde siempre en español natural, con rigor agronómico y una extensión mínima de 320 caracteres "
        "(entre 350 y 650 caracteres) en un texto coherente y directamente aplicable."
    )

    user_prompt = (
        f"Diagnóstico CNN de hoja de papa:\n"
        f"- Identificador de clase: {disease_id}\n"
        f"- Diagnóstico: {disease_name} ({scientific_name})\n"
        f"- Confianza del modelo: {confidence_pct:.2f}%\n"
        f"- Nivel de severidad: {severity}\n"
        f"- Distribución de probabilidades: {breakdown_text}\n"
        f"- Estrategia seleccionada por el agente RL: {clean_strategy}\n\n"
        "Genera una recomendación agronómica en español adaptada exactamente a este diagnóstico y estrategia, "
        "asegurando más de 250 caracteres con medidas claras, seguras y accionables."
    )
    return system_prompt, user_prompt


_FALLBACK_TEMPLATES: dict[str, dict[str, str]] = {
    "Potato___Early_blight": {
        "tecnico": (
            "Protocolo técnico para Tizón Temprano (Alternaria solani): Se recomienda iniciar aplicaciones preventivas o "
            "curativas tempranas con fungicidas protectantes de contacto como Mancozeb o Clorotalonil (grupos FRAC M03/M05) "
            "en rotación con estrobilurinas o triazoles si la presión de inóculo aumenta, respetando intervalos de 7 a 10 días "
            "y calibración de boquillas para cobertura uniforme del tercio inferior. Reduzca el mojado foliar evitando riego por "
            "aspersión vespertino, retire residuos senescentes y verifique niveles adecuados de potasio y nitrógeno."
        ),
        "practico": (
            "Guía práctica para controlar el Tizón Temprano en tu cultivo: Revisa primero las hojas más bajas de la planta y "
            "retira con cuidado las que tengan manchas oscuras con anillos, sacándolas del lote en una bolsa para no regar las "
            "esporas. Riega temprano en la mañana directo al suelo sin mojar las hojas, desinfecta tijeras y herramientas después "
            "de trabajar en las plantas afectadas y aplica un fungicida recomendado para papa cubriendo bien el envés foliar."
        ),
        "ecologico": (
            "Manejo agroecológico para Tizón Temprano (Alternaria solani): Realiza poda sanitaria manual de las hojas basales "
            "afectadas y retíralas del cultivo para cortar el ciclo del hongo. Mejora la ventilación entre surcos mediante un "
            "aporque oportuno y aplica bioinsumos basados en Trichoderma harzianum o Bacillus subtilis, alternando con caldo "
            "bordelés o extractos de cola de caballo al atardecer. Refuerza el suelo con compost maduro y planifica rotación de "
            "al menos dos ciclos con cultivos no solanáceos."
        ),
    },
    "Potato___Late_blight": {
        "tecnico": (
            "Protocolo técnico prioritario para Tizón Tardío (Phytophthora infestans): Dada la severidad crítica de este "
            "oomiceto, aplique de inmediato fungicidas sistémicos y translaminares específicos como Metalaxil-M, Cimoxanilo o "
            "Dimetomorf en mezcla con protectante multisitio (Mancozeb o Fluazinam) rotando códigos FRAC para evitar resistencia. "
            "Elimine focos primarios con alta humedad relativa, suspenda el riego por aspersión y monitoree cada 48 a 72 horas "
            "las condiciones de temperatura fresca y rocío persistente en el follaje."
        ),
        "practico": (
            "Acción urgente en campo contra el Tizón Tardío: Esta enfermedad avanza muy rápido con frío y humedad, por lo que "
            "debes actuar hoy mismo. Recorre el lote, marca los focos con manchas húmedas oscuras y elimina las plantas muy "
            "dañadas para frenar el contagio por el viento. Aplica de inmediato un fungicida sistémico específico para gota o "
            "tizón tardío siguiendo la dosis de la etiqueta, evita mojar las hojas al regar y revisa el cultivo cada dos días."
        ),
        "ecologico": (
            "Manejo agroecológico de contención para Tizón Tardío (Phytophthora infestans): Retira de forma inmediata las hojas y "
            "tallos con lesiones acuosas y destrúyelos fuera del lote para frenar la dispersión de esporangios. Favorece el "
            "secado rápido del follaje mejorando el drenaje y la aireación entre surcos. Aplica preventivamente preparados de "
            "cobre en dosis orgánicas autorizadas combinados con bioestimulantes y antagonistas como Bacillus subtilis, y "
            "utiliza semilla certificada tolerante en futuras siembras."
        ),
    },
    "Potato___healthy": {
        "tecnico": (
            "Plan técnico preventivo para Planta Sana (Solanum tuberosum): El follaje muestra vigor y ausencia de lesiones "
            "patogénicas activas. Mantenga un programa de fertilización balanceado evitando excesos de nitrógeno que generan "
            "tejido suculento susceptible a oomicetos y hongos foliares. Implemente monitoreo fitosanitario semanal mediante "
            "muestreos en zigzag, registre variables de humedad relativa y temperatura, verifique el drenaje de los surcos y "
            "conserve el control preventivo de insectos vectores."
        ),
        "practico": (
            "Cuidados prácticos para mantener tu cultivo sano: ¡Buenas noticias! La hoja analizada no muestra señales de "
            "enfermedad. Para conservar el lote saludable, sigue revisando tus plantas al menos una vez por semana, "
            "especialmente después de lluvias o noches húmedas. Mantén los surcos libres de malezas para que circule el aire, "
            "riega de forma equilibrada sin encharcar la raíz y aplica el abono recomendado sin excederte en nitrógeno."
        ),
        "ecologico": (
            "Manejo agroecológico preventivo para Planta Sana: Tu planta presenta un estado fitosanitario óptimo. Fortalece su "
            "resiliencia natural incorporando microorganismos benéficos al suelo como Trichoderma spp. y micorrizas, junto con "
            "biofertilizantes ricos en potasio y silicio natural. Conserva coberturas vegetales y bordes con flores benéficas "
            "para atraer controladores biológicos de insectos vectores, mantén buen drenaje y realiza recorridos semanales."
        ),
    },
}


def build_structured_fallback(
    disease_id: str,
    disease_name: str,
    scientific_name: str,
    confidence_pct: float,
    severity: str,
    breakdown: list[dict[str, Any]],
    strategy: str,
) -> str:
    """Genera una recomendación determinística en español garantizada > 200 caracteres."""
    clean_strategy = _normalize_strategy(strategy)
    strategy_templates = _FALLBACK_TEMPLATES.get(disease_id)

    if strategy_templates and clean_strategy in strategy_templates:
        base_text = strategy_templates[clean_strategy]
    else:
        base_text = (
            f"Recomendación agronómica ({clean_strategy}) para {disease_name} ({scientific_name}): "
            f"El sistema registró una severidad '{severity}'. Se sugiere inspeccionar el follaje del lote, "
            "aislar o retirar tejido con síntomas visibles, evitar el exceso de humedad foliar durante el riego, "
            "desinfectar herramientas agrícolas entre surcos y mantener un seguimiento fitosanitario periódico "
            "junto con rotación de cultivos y manejo nutricional equilibrado."
        )

    breakdown_summary = _format_breakdown_summary(breakdown)
    context_line = (
        f" [Diagnóstico CNN: {disease_name} con {confidence_pct:.2f}% de confianza; "
        f"severidad: {severity}; probabilidades: {breakdown_summary}.]"
    )
    full_text = f"{base_text}{context_line}".strip()
    if len(full_text) <= MIN_RECOMMENDATION_CHARS:
        full_text += (
            " Mantenga registro de campo tras cada intervención y ajuste las prácticas de manejo "
            "integrado según la evolución climática del cultivo de papa."
        )
    return full_text


def _extract_gemini_text(response: Any) -> str:
    try:
        direct_text = getattr(response, "text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()
    except ValueError:
        pass

    candidates = getattr(response, "candidates", None) or []
    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text_val = getattr(part, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                parts_text.append(text_val.strip())
        if parts_text:
            break
    return "\n".join(parts_text).strip()


def _call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=15_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=700,
        ),
    )
    return _extract_gemini_text(response)


def _call_deepseek(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=20.0,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=700,
        stream=False,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content.strip() if isinstance(content, str) else ""


def generate_recommendation(
    disease_id: str,
    disease_name: str,
    scientific_name: str,
    confidence: float,
    severity: str,
    breakdown: list[dict[str, Any]],
    strategy: str,
) -> dict[str, Any]:
    """
    Genera una recomendación agronómica en español (> 200 caracteres garantizados)
    usando Google Gemini como proveedor primario por defecto, DeepSeek configurable
    vía .env, y un fallback estructurado determinístico ante ausencia de clave o fallos.
    """
    clean_strategy = _normalize_strategy(strategy)
    confidence_pct = _normalize_confidence_pct(confidence)
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower() or "gemini"

    system_prompt, user_prompt = build_prompts(
        disease_id=disease_id,
        disease_name=disease_name,
        scientific_name=scientific_name,
        confidence_pct=confidence_pct,
        severity=severity,
        breakdown=breakdown,
        strategy=clean_strategy,
    )

    fallback_text = build_structured_fallback(
        disease_id=disease_id,
        disease_name=disease_name,
        scientific_name=scientific_name,
        confidence_pct=confidence_pct,
        severity=severity,
        breakdown=breakdown,
        strategy=clean_strategy,
    )

    raw_text = ""
    used_provider = "fallback"
    used_model = f"structured-{clean_strategy}"
    is_fallback = True
    fallback_reason: Optional[str] = None

    if provider == "gemini":
        gemini_model = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not _is_valid_api_key(gemini_key):
            fallback_reason = "GEMINI_API_KEY no configurada o inválida; usando fallback estructurado."
        else:
            candidate_models: list[str] = [gemini_model]
            for alt_model in GEMINI_FALLBACK_MODELS:
                if alt_model not in candidate_models:
                    candidate_models.append(alt_model)

            last_exc: Optional[Exception] = None
            for candidate_model in candidate_models:
                try:
                    raw_text = _call_gemini(gemini_key, candidate_model, system_prompt, user_prompt)
                    if raw_text:
                        used_provider = "gemini"
                        used_model = candidate_model
                        is_fallback = False
                        fallback_reason = None
                        break
                    else:
                        fallback_reason = "Respuesta vacía de Gemini; usando fallback estructurado."
                except Exception as exc:
                    last_exc = exc
                    continue

            if is_fallback and last_exc is not None:
                fallback_reason = f"Error al consultar Gemini ({type(last_exc).__name__}); usando fallback estructurado."

    elif provider == "deepseek":
        deepseek_model = (os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL).strip() or DEFAULT_DEEPSEEK_MODEL
        deepseek_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not _is_valid_api_key(deepseek_key):
            fallback_reason = "DEEPSEEK_API_KEY no configurada o inválida; usando fallback estructurado."
        else:
            try:
                raw_text = _call_deepseek(deepseek_key, deepseek_model, system_prompt, user_prompt)
                if raw_text:
                    used_provider = "deepseek"
                    used_model = deepseek_model
                    is_fallback = False
                else:
                    fallback_reason = "Respuesta vacía de DeepSeek; usando fallback estructurado."
            except Exception as exc:
                fallback_reason = f"Error al consultar DeepSeek ({type(exc).__name__}); usando fallback estructurado."
    else:
        fallback_reason = f"Proveedor LLM no soportado ('{provider}'); usando fallback estructurado."

    if not is_fallback:
        cleaned_text = raw_text.strip()
        if len(cleaned_text) <= MIN_RECOMMENDATION_CHARS:
            # Garantía estricta > 200 caracteres si el LLM devolvió una respuesta muy breve
            recommendation = f"{cleaned_text}\n\n{fallback_text}".strip()
            fallback_reason = "Respuesta del proveedor <= 200 caracteres; complementada con guía estructurada."
        else:
            recommendation = cleaned_text
    else:
        recommendation = fallback_text

    # Invariante final absoluta
    if len(recommendation.strip()) <= MIN_RECOMMENDATION_CHARS:
        recommendation = fallback_text

    return {
        "recommendation": recommendation,
        "provider": used_provider,
        "model": used_model,
        "strategy": clean_strategy,
        "char_count": len(recommendation),
        "is_fallback": is_fallback,
        "fallback_reason": fallback_reason,
    }


def chat_followup(
    disease_id: str,
    disease_name: str,
    scientific_name: str,
    confidence: float,
    severity: str,
    strategy: str,
    initial_recommendation: str,
    history: list[dict[str, str]],
    user_message: str,
) -> dict[str, Any]:
    """
    Continúa la conversación con el usuario manteniendo el contexto del diagnóstico CNN,
    la recomendación inicial generada y la estrategia activa (tecnico, practico, ecologico).
    """
    clean_strategy = _normalize_strategy(strategy)
    confidence_pct = _normalize_confidence_pct(confidence)
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower() or "gemini"
    clean_question = (user_message or "").strip()

    system_prompt, _ = build_prompts(
        disease_id=disease_id,
        disease_name=disease_name,
        scientific_name=scientific_name,
        confidence_pct=confidence_pct,
        severity=severity,
        breakdown=[],
        strategy=clean_strategy,
    )
    chat_system_prompt = (
        f"{system_prompt}\n"
        "Estás en una conversación de seguimiento con el usuario sobre una hoja de papa recién diagnosticada. "
        "Puedes usar formato Markdown claro (negritas, listas con viñetas, pasos numerados) en español para que "
        "la respuesta sea fácil de leer. Responde directamente a la duda del usuario manteniendo el enfoque de "
        f"la estrategia '{clean_strategy}'."
    )

    # Construir transcripción compacta del contexto e historial reciente
    history_lines: list[str] = []
    for turn in (history or [])[-8:]:
        if not isinstance(turn, dict):
            continue
        role = "Usuario" if turn.get("role") == "user" else "AgroScan AI"
        content = (turn.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "Sin mensajes previos adicionales."

    composed_user_prompt = (
        f"Contexto del diagnóstico actual:\n"
        f"- Patología: {disease_name} ({scientific_name}) [ID: {disease_id}]\n"
        f"- Confianza del modelo: {confidence_pct:.2f}%\n"
        f"- Severidad: {severity}\n"
        f"- Estrategia activa: {clean_strategy}\n"
        f"- Recomendación inicial entregada:\n{initial_recommendation}\n\n"
        f"Historial de la conversación:\n{history_block}\n\n"
        f"Nueva pregunta del usuario:\n{clean_question}\n\n"
        "Responde en español con formato Markdown claro y útil."
    )

    fallback_reply = (
        f"**Seguimiento ({clean_strategy.capitalize()}) para {disease_name} (*{scientific_name}*):**\n\n"
        f"Respecto a tu consulta (*\"{clean_question}\"*):\n"
        f"- **Monitoreo continuo:** Revisa el tercio inferior y medio del follaje cada 48 a 72 horas para evaluar la evolución de síntomas.\n"
        f"- **Manejo de humedad:** Evita el riego foliar en horas de la tarde o noche para reducir la germinación de esporas.\n"
        f"- **Aplicación segura:** Respeta siempre la dosis técnica de la etiqueta, el equipo de protección personal y la rotación de modos de acción según la severidad ({severity})."
    )

    raw_text = ""
    used_provider = "fallback"
    used_model = f"structured-{clean_strategy}"
    is_fallback = True
    fallback_reason: Optional[str] = None

    if provider == "gemini":
        gemini_model = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not _is_valid_api_key(gemini_key):
            fallback_reason = "GEMINI_API_KEY no configurada o inválida; usando respuesta estructurada."
        else:
            candidate_models: list[str] = [gemini_model]
            for alt_model in GEMINI_FALLBACK_MODELS:
                if alt_model not in candidate_models:
                    candidate_models.append(alt_model)

            last_exc: Optional[Exception] = None
            for candidate_model in candidate_models:
                try:
                    raw_text = _call_gemini(gemini_key, candidate_model, chat_system_prompt, composed_user_prompt)
                    if raw_text:
                        used_provider = "gemini"
                        used_model = candidate_model
                        is_fallback = False
                        fallback_reason = None
                        break
                except Exception as exc:
                    last_exc = exc
                    continue

            if is_fallback and last_exc is not None:
                fallback_reason = f"Error al consultar Gemini ({type(last_exc).__name__}); usando respuesta estructurada."

    elif provider == "deepseek":
        deepseek_model = (os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL).strip() or DEFAULT_DEEPSEEK_MODEL
        deepseek_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not _is_valid_api_key(deepseek_key):
            fallback_reason = "DEEPSEEK_API_KEY no configurada o inválida; usando respuesta estructurada."
        else:
            try:
                raw_text = _call_deepseek(deepseek_key, deepseek_model, chat_system_prompt, composed_user_prompt)
                if raw_text:
                    used_provider = "deepseek"
                    used_model = deepseek_model
                    is_fallback = False
            except Exception as exc:
                fallback_reason = f"Error al consultar DeepSeek ({type(exc).__name__}); usando respuesta estructurada."

    reply = raw_text.strip() if (not is_fallback and raw_text.strip()) else fallback_reply

    return {
        "reply": reply,
        "provider": used_provider,
        "model": used_model,
        "strategy": clean_strategy,
        "char_count": len(reply),
        "is_fallback": is_fallback,
        "fallback_reason": fallback_reason,
    }
