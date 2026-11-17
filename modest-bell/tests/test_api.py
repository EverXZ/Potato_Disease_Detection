import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import app as app_module
from rl_agent import ACTIONS, STATES, ContextualBanditAgent


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(autouse=True)
def isolated_rl_and_offline_llm(tmp_path: Path, monkeypatch):
    """Aísla rl_policy.json en tmp_path para nunca ensuciar data/rl_policy.json."""
    temp_agent = ContextualBanditAgent(policy_path=tmp_path / "rl_policy.json", epsilon=0.2, alpha=0.2, seed=42)
    monkeypatch.setattr(app_module, "rl_agent", temp_agent)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return temp_agent


def _make_png_bytes(color=(34, 139, 34)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_home_and_model_status(client: TestClient):
    resp_home = client.get("/")
    assert resp_home.status_code == 200
    html = resp_home.text
    assert "Panel de Métricas RL en Vivo" in html
    assert "Califica esta recomendación" in html
    assert "/feedback" in html
    assert "/rl-stats" in html

    resp_status = client.get("/model-status")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    assert "has_model" in status_data
    assert "keras_available" in status_data


def test_rl_stats_initial(client: TestClient):
    resp = client.get("/rl-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert tuple(data["states"]) == STATES
    assert tuple(data["actions"]) == ACTIONS
    assert data["total_feedbacks"] == 0
    assert data["average_reward"] == 0.0
    assert data["epsilon"] == pytest.approx(0.2)


def test_predict_returns_llm_and_rl_fields(client: TestClient):
    img_buf = _make_png_bytes()
    resp = client.post("/predict", files={"file": ("leaf.png", img_buf, "image/png")})
    assert resp.status_code == 200

    data = resp.json()
    assert data["success"] is True
    assert data["prediction"]["disease_id"] in STATES
    assert "disease_name" in data["prediction"]
    assert "confidence" in data["prediction"]
    assert len(data["breakdown"]) == 3

    assert isinstance(data["llm_recommendation"], str)
    assert len(data["llm_recommendation"].strip()) > 200
    assert data["llm_info"]["char_count"] == len(data["llm_recommendation"]) > 200
    assert data["llm_info"]["strategy"] in ACTIONS

    assert data["rl_info"]["state"] == data["prediction"]["disease_id"]
    assert data["rl_info"]["strategy"] in ACTIONS
    assert data["rl_info"]["mode"] in ("exploration", "exploitation")
    assert data["rl_info"]["mode_label"] in ("Exploración", "Explotación")


def test_predict_rejects_non_image(client: TestClient):
    resp = client.post("/predict", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_feedback_updates_q_table_and_persists(client: TestClient, isolated_rl_and_offline_llm: ContextualBanditAgent):
    payload = {
        "disease_id": "Potato___Late_blight",
        "strategy": "tecnico",
        "rating": 5,
        "comment": "Protocolo muy claro y útil",
    }
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["update"]["reward"] == pytest.approx(1.0)
    assert body["update"]["q_before"] == pytest.approx(0.0)
    assert body["update"]["q_after"] == pytest.approx(0.2)
    assert body["stats"]["total_feedbacks"] == 1
    assert body["stats"]["q_table"]["Potato___Late_blight"]["tecnico"] == pytest.approx(0.2)

    # Verificar que GET /rl-stats refleja el cambio y existe persistencia en el archivo temporal
    stats_resp = client.get("/rl-stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_feedbacks"] == 1
    assert stats["average_reward"] == pytest.approx(1.0)
    assert stats["counts"]["Potato___Late_blight"]["tecnico"] == 1
    assert isolated_rl_and_offline_llm.policy_path.exists()


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"disease_id": "Potato___healthy", "strategy": "tecnico", "rating": 0},
        {"disease_id": "Potato___healthy", "strategy": "tecnico", "rating": 6},
        {"disease_id": "Potato___healthy", "strategy": "tecnico", "rating": True},
        {"disease_id": "Potato___healthy", "strategy": "desconocida", "rating": 5},
        {"disease_id": "   ", "strategy": "tecnico", "rating": 5},
    ],
)
def test_feedback_validation_rejects_invalid_inputs(client: TestClient, bad_payload: dict):
    resp = client.post("/feedback", json=bad_payload)
    assert resp.status_code in (400, 422)


def test_feedback_validation_rejects_overly_long_fields_with_422(client: TestClient):
    resp_long_disease = client.post(
        "/feedback",
        json={
            "disease_id": "x" * 129,
            "strategy": "tecnico",
            "rating": 5,
            "comment": "ok",
        },
    )
    assert resp_long_disease.status_code == 422

    resp_long_comment = client.post(
        "/feedback",
        json={
            "disease_id": "Potato___healthy",
            "strategy": "tecnico",
            "rating": 5,
            "comment": "c" * 501,
        },
    )
    assert resp_long_comment.status_code == 422


def test_regenerate_and_chat_endpoints(client):
    regen_resp = client.post(
        "/regenerate",
        json={
            "disease_id": "Potato___Late_blight",
            "disease_name": "Tizón Tardío",
            "scientific_name": "Phytophthora infestans",
            "confidence": 92.5,
            "severity": "Crítica / Muy Alta",
            "breakdown": [],
            "strategy": "ecologico",
        },
    )
    assert regen_resp.status_code == 200
    regen_data = regen_resp.json()
    assert regen_data["success"] is True
    assert regen_data["rl_info"]["strategy"] == "ecologico"
    assert regen_data["rl_info"]["mode"] == "manual"
    assert len(regen_data["llm_recommendation"]) > 200

    chat_resp = client.post(
        "/chat",
        json={
            "disease_id": "Potato___Late_blight",
            "disease_name": "Tizón Tardío",
            "scientific_name": "Phytophthora infestans",
            "confidence": 92.5,
            "severity": "Crítica / Muy Alta",
            "strategy": "ecologico",
            "initial_recommendation": regen_data["llm_recommendation"],
            "history": [],
            "message": "¿Cada cuántos días aplico el tratamiento?",
        },
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["success"] is True
    assert len(chat_data["reply"]) > 50
