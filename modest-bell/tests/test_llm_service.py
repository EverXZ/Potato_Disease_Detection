import sys
import types as py_types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import llm_service


SAMPLE_BREAKDOWN = [
    {"class_idx": 1, "disease_id": "Potato___Late_blight", "name": "Tizón Tardío", "probability": 88.5, "color": "red"},
    {"class_idx": 0, "disease_id": "Potato___Early_blight", "name": "Tizón Temprano", "probability": 9.5, "color": "amber"},
    {"class_idx": 2, "disease_id": "Potato___healthy", "name": "Planta Sana", "probability": 2.0, "color": "emerald"},
]


@pytest.mark.parametrize(
    "disease_id,disease_name,sci_name,severity",
    [
        ("Potato___Early_blight", "Tizón Temprano", "Alternaria solani", "Moderada / Alta"),
        ("Potato___Late_blight", "Tizón Tardío", "Phytophthora infestans", "Crítica / Muy Alta"),
        ("Potato___healthy", "Planta Sana", "Solanum tuberosum (Saludable)", "Ninguna"),
    ],
)
@pytest.mark.parametrize("strategy", ["tecnico", "practico", "ecologico"])
def test_fallback_all_classes_and_strategies_over_200_chars(
    monkeypatch,
    disease_id: str,
    disease_name: str,
    sci_name: str,
    severity: str,
    strategy: str,
):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    res = llm_service.generate_recommendation(
        disease_id=disease_id,
        disease_name=disease_name,
        scientific_name=sci_name,
        confidence=91.25,
        severity=severity,
        breakdown=SAMPLE_BREAKDOWN,
        strategy=strategy,
    )

    assert res["provider"] == "fallback"
    assert res["is_fallback"] is True
    assert res["strategy"] == strategy
    assert len(res["recommendation"].strip()) > 200
    assert res["char_count"] == len(res["recommendation"])
    assert res["char_count"] > 200


def test_prompts_are_distinct_per_strategy():
    p_tec = llm_service.build_prompts(
        "Potato___Early_blight", "Tizón Temprano", "Alternaria solani", 90.0, "Moderada / Alta", SAMPLE_BREAKDOWN, "tecnico"
    )
    p_pra = llm_service.build_prompts(
        "Potato___Early_blight", "Tizón Temprano", "Alternaria solani", 90.0, "Moderada / Alta", SAMPLE_BREAKDOWN, "practico"
    )
    p_eco = llm_service.build_prompts(
        "Potato___Early_blight", "Tizón Temprano", "Alternaria solani", 90.0, "Moderada / Alta", SAMPLE_BREAKDOWN, "ecologico"
    )
    assert p_tec[0] != p_pra[0] != p_eco[0]
    assert "FRAC" in p_tec[0]
    assert "agricultor" in p_pra[0]
    assert "agroecología" in p_eco[0]


def test_confidence_normalization_fraction_vs_percent(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res_frac = llm_service.generate_recommendation(
        "Potato___healthy", "Planta Sana", "Solanum tuberosum", 0.92, "Ninguna", SAMPLE_BREAKDOWN, "practico"
    )
    assert "92.00%" in res_frac["recommendation"]


def test_mocked_gemini_success_and_valueerror_candidates(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345678")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    long_spanish_response = (
        "Recomendación técnica detallada para el control del tizón tardío en papa: aplique de inmediato un fungicida "
        "sistémico con Metalaxil-M más Mancozeb rotando grupos FRAC, inspeccione el envés foliar cada 48 horas, "
        "suspenda temporalmente el riego por aspersión y retire los focos primarios del cultivo para cortar la esporulación."
    )
    assert len(long_spanish_response) > 200

    class FakeModels:
        def generate_content(self, model, contents, config):
            return py_types.SimpleNamespace(text=long_spanish_response)

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            assert api_key == "fake-gemini-key-12345678"
            self.models = FakeModels()

    import google.genai as genai_mod

    monkeypatch.setattr(genai_mod, "Client", FakeClient)

    res = llm_service.generate_recommendation(
        "Potato___Late_blight", "Tizón Tardío", "Phytophthora infestans", 88.5, "Crítica / Muy Alta", SAMPLE_BREAKDOWN, "tecnico"
    )
    assert res["provider"] == "gemini"
    assert res["model"] == "gemini-2.5-flash"
    assert res["is_fallback"] is False
    assert len(res["recommendation"]) > 200

    # Caso donde response.text lanza ValueError pero candidates[0].content.parts tiene texto
    class CandidateResponse:
        @property
        def text(self):
            raise ValueError("Non-text parts present")

        candidates = [
            py_types.SimpleNamespace(
                content=py_types.SimpleNamespace(parts=[py_types.SimpleNamespace(text=long_spanish_response)])
            )
        ]

    class FakeModelsCand:
        def generate_content(self, model, contents, config):
            return CandidateResponse()

    class FakeClientCand:
        def __init__(self, api_key, http_options=None):
            self.models = FakeModelsCand()

    monkeypatch.setattr(genai_mod, "Client", FakeClientCand)
    res_cand = llm_service.generate_recommendation(
        "Potato___Late_blight", "Tizón Tardío", "Phytophthora infestans", 88.5, "Crítica / Muy Alta", SAMPLE_BREAKDOWN, "tecnico"
    )
    assert res_cand["provider"] == "gemini"
    assert res_cand["is_fallback"] is False
    assert len(res_cand["recommendation"]) > 200


def test_mocked_short_response_is_guaranteed_over_200(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345678")

    class FakeModelsShort:
        def generate_content(self, model, contents, config):
            return py_types.SimpleNamespace(text="Aplica fungicida hoy mismo.")

    class FakeClientShort:
        def __init__(self, api_key, http_options=None):
            self.models = FakeModelsShort()

    import google.genai as genai_mod

    monkeypatch.setattr(genai_mod, "Client", FakeClientShort)

    res = llm_service.generate_recommendation(
        "Potato___Early_blight", "Tizón Temprano", "Alternaria solani", 85.0, "Moderada / Alta", SAMPLE_BREAKDOWN, "practico"
    )
    assert len(res["recommendation"].strip()) > 200
    assert res["char_count"] == len(res["recommendation"]) > 200


def test_mocked_deepseek_success_and_error_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key-12345678")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    long_ds_text = (
        "Manejo agroecológico integral para tizón temprano en papa: pode las hojas basales con anillos necróticos, "
        "retírelas del lote en bolsa cerrada, mejore la aireación entre surcos, aplique Trichoderma harzianum al "
        "atardecer y refuerce con caldo bordelés autorizado y rotación con especies no solanáceas."
    )

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "deepseek-chat"
            return py_types.SimpleNamespace(
                choices=[py_types.SimpleNamespace(message=py_types.SimpleNamespace(content=long_ds_text))]
            )

    class FakeOpenAI:
        def __init__(self, api_key, base_url, timeout):
            assert api_key == "fake-deepseek-key-12345678"
            assert base_url == "https://api.deepseek.com"
            self.chat = py_types.SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    res = llm_service.generate_recommendation(
        "Potato___Early_blight", "Tizón Temprano", "Alternaria solani", 80.0, "Moderada / Alta", SAMPLE_BREAKDOWN, "ecologico"
    )
    assert res["provider"] == "deepseek"
    assert res["model"] == "deepseek-chat"
    assert res["is_fallback"] is False
    assert len(res["recommendation"]) > 200

    # Si DeepSeek falla por red/timeout -> fallback > 200
    class FailingOpenAI:
        def __init__(self, api_key, base_url, timeout):
            raise RuntimeError("Timeout simulado")

    monkeypatch.setattr(openai, "OpenAI", FailingOpenAI)
    res_err = llm_service.generate_recommendation(
        "Potato___Early_blight", "Tizón Temprano", "Alternaria solani", 80.0, "Moderada / Alta", SAMPLE_BREAKDOWN, "ecologico"
    )
    assert res_err["provider"] == "fallback"
    assert res_err["is_fallback"] is True
    assert len(res_err["recommendation"]) > 200


def test_format_breakdown_summary_percentage_vs_fraction_scales():
    pct_breakdown = [
        {"name": "Tizón Tardío", "probability": 99.2},
        {"name": "Tizón Temprano", "probability": 0.5},
        {"name": "Planta Sana", "probability": 0.3},
    ]
    pct_summary = llm_service._format_breakdown_summary(pct_breakdown)
    assert pct_summary == "Tizón Tardío: 99.20% | Tizón Temprano: 0.50% | Planta Sana: 0.30%"
    assert "50.00%" not in pct_summary
    assert "30.00%" not in pct_summary

    frac_breakdown = [
        {"name": "Tizón Tardío", "probability": 0.8},
        {"name": "Tizón Temprano", "probability": 0.15},
        {"name": "Planta Sana", "probability": 0.05},
    ]
    frac_summary = llm_service._format_breakdown_summary(frac_breakdown)
    assert frac_summary == "Tizón Tardío: 80.00% | Tizón Temprano: 15.00% | Planta Sana: 5.00%"


def test_whitespace_llm_provider_defaults_to_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "   ")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345678")

    long_spanish_response = (
        "Recomendación técnica para el manejo del tizón temprano en el cultivo de papa: "
        "inicie aplicaciones preventivas con Mancozeb o Clorotalonil respetando la rotación "
        "FRAC, reduzca el mojado foliar por riego y retire hojas basales senescentes."
    )

    class FakeModels:
        def generate_content(self, model, contents, config):
            return py_types.SimpleNamespace(text=long_spanish_response)

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.models = FakeModels()

    import google.genai as genai_mod

    monkeypatch.setattr(genai_mod, "Client", FakeClient)

    res = llm_service.generate_recommendation(
        "Potato___Early_blight", "Tizón Temprano", "Alternaria solani", 99.2, "Moderada / Alta", SAMPLE_BREAKDOWN, "tecnico"
    )
    assert res["provider"] == "gemini"
    assert res["is_fallback"] is False
    assert len(res["recommendation"]) > 200
