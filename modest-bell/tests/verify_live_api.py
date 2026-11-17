"""
Verificación condicional contra la API real de Gemini o DeepSeek.
Si existen claves reales configuradas en .env o variables de entorno, ejecuta
una llamada real y verifica que devuelva > 200 caracteres con is_fallback == False.
Si no hay claves configuradas, reporta explícitamente sin fallar.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False, encoding="utf-8")

import llm_service


def main() -> int:
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    deepseek_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()

    has_gemini = llm_service._is_valid_api_key(gemini_key)
    has_deepseek = llm_service._is_valid_api_key(deepseek_key)

    if provider == "gemini" and not has_gemini:
        print("LIVE_API_SKIPPED: No hay GEMINI_API_KEY válida configurada en .env ni en el entorno.")
        return 0
    if provider == "deepseek" and not has_deepseek:
        print("LIVE_API_SKIPPED: No hay DEEPSEEK_API_KEY válida configurada en .env ni en el entorno.")
        return 0
    if provider not in ("gemini", "deepseek"):
        print(f"LIVE_API_SKIPPED: Proveedor '{provider}' no reconocido para verificación en vivo.")
        return 0

    breakdown = [
        {"class_idx": 1, "disease_id": "Potato___Late_blight", "name": "Tizón Tardío", "probability": 91.5, "color": "red"},
        {"class_idx": 0, "disease_id": "Potato___Early_blight", "name": "Tizón Temprano", "probability": 6.5, "color": "amber"},
        {"class_idx": 2, "disease_id": "Potato___healthy", "name": "Planta Sana", "probability": 2.0, "color": "emerald"},
    ]

    result = llm_service.generate_recommendation(
        disease_id="Potato___Late_blight",
        disease_name="Tizón Tardío",
        scientific_name="Phytophthora infestans",
        confidence=91.5,
        severity="Crítica / Muy Alta",
        breakdown=breakdown,
        strategy="tecnico",
    )

    print(
        f"LIVE_API_RESULT: provider={result['provider']} model={result['model']} "
        f"is_fallback={result['is_fallback']} char_count={result['char_count']}"
    )
    if result["is_fallback"] or result["char_count"] <= 200:
        print(f"LIVE_API_ERROR: fallback_reason={result.get('fallback_reason')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
