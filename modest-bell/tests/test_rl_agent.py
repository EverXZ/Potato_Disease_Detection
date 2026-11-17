import json
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl_agent import ACTIONS, STATES, ContextualBanditAgent


def test_initial_states_and_stats(tmp_path: Path):
    policy_path = tmp_path / "rl_policy.json"
    agent = ContextualBanditAgent(policy_path=policy_path, epsilon=0.2, alpha=0.2, seed=42)
    stats = agent.get_stats()

    assert tuple(stats["states"]) == STATES
    assert tuple(stats["actions"]) == ACTIONS
    assert stats["total_feedbacks"] == 0
    assert stats["average_reward"] == 0.0
    for state in STATES:
        for action in ACTIONS:
            assert stats["q_table"][state][action] == 0.0
            assert stats["counts"][state][action] == 0


def test_reward_mapping_and_q_update_math(tmp_path: Path):
    policy_path = tmp_path / "rl_policy.json"
    agent = ContextualBanditAgent(policy_path=policy_path, epsilon=0.0, alpha=0.5, seed=1)

    expected_rewards = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
    for idx, (rating, expected_r) in enumerate(expected_rewards.items(), start=1):
        res = agent.update("Potato___healthy", "tecnico", rating)
        assert res["reward"] == pytest.approx(expected_r)
        assert res["total_feedbacks"] == idx

    # Verificar convergencia de Q(s, a) <- Q(s, a) + alpha * (r - Q(s, a))
    agent2 = ContextualBanditAgent(policy_path=tmp_path / "conv.json", epsilon=0.0, alpha=0.25, seed=7)
    r1 = agent2.update("Potato___Late_blight", "ecologico", 5, comment="Excelente manejo orgánico")
    assert r1["q_before"] == pytest.approx(0.0)
    assert r1["q_after"] == pytest.approx(0.25)

    r2 = agent2.update("Potato___Late_blight", "ecologico", 5)
    # 0.25 + 0.25 * (1.0 - 0.25) = 0.4375
    assert r2["q_after"] == pytest.approx(0.4375)

    for _ in range(25):
        agent2.update("Potato___Late_blight", "ecologico", 5)
    assert agent2.get_stats()["q_table"]["Potato___Late_blight"]["ecologico"] > 0.99


def test_epsilon_greedy_exploitation_and_tie_breaking(tmp_path: Path):
    # Con Q iguales a 0.0, la explotación debe romper empates aleatoriamente (no bloquearse en "tecnico")
    agent_ties = ContextualBanditAgent(policy_path=tmp_path / "ties.json", epsilon=0.0, alpha=0.2, seed=123)
    picked = {agent_ties.select_action("Potato___Early_blight")["strategy"] for _ in range(30)}
    assert len(picked) > 1

    # Después de recompensar "ecologico", epsilon=0.0 debe explotar "ecologico" determinísticamente
    agent_ties.update("Potato___Early_blight", "ecologico", 5)
    for _ in range(10):
        choice = agent_ties.select_action("Potato___Early_blight")
        assert choice["strategy"] == "ecologico"
        assert choice["mode"] == "exploitation"
        assert choice["mode_label"] == "Explotación"
        assert choice["is_exploration"] is False


def test_pure_exploration_mode(tmp_path: Path):
    agent = ContextualBanditAgent(policy_path=tmp_path / "explore.json", epsilon=1.0, alpha=0.2, seed=9)
    agent.update("Potato___healthy", "tecnico", 5)
    seen = set()
    for _ in range(40):
        choice = agent.select_action("Potato___healthy")
        assert choice["mode"] == "exploration"
        assert choice["mode_label"] == "Exploración"
        assert choice["is_exploration"] is True
        seen.add(choice["strategy"])
    assert seen == set(ACTIONS)


def test_persistence_utf8_and_corrupt_recovery(tmp_path: Path):
    policy_path = tmp_path / "nested" / "rl_policy.json"
    agent = ContextualBanditAgent(policy_path=policy_path, epsilon=0.2, alpha=0.3, seed=42)
    agent.update("Potato___Early_blight", "practico", 4, comment="Muy útil en campo agrícola")

    assert policy_path.exists()
    raw = json.loads(policy_path.read_text(encoding="utf-8"))
    assert raw["total_feedbacks"] == 1
    assert raw["recent_feedback"][0]["comment"] == "Muy útil en campo agrícola"

    # Recargar en nueva instancia
    reloaded = ContextualBanditAgent(policy_path=policy_path, epsilon=0.2, alpha=0.3, seed=42)
    stats = reloaded.get_stats()
    assert stats["total_feedbacks"] == 1
    assert stats["q_table"]["Potato___Early_blight"]["practico"] == pytest.approx(0.15)
    assert stats["counts"]["Potato___Early_blight"]["practico"] == 1

    # Recuperación ante JSON corrupto o vacío
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{corrupto", encoding="utf-8")
    recovered = ContextualBanditAgent(policy_path=corrupt_path)
    assert recovered.get_stats()["total_feedbacks"] == 0

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("", encoding="utf-8")
    recovered_empty = ContextualBanditAgent(policy_path=empty_path)
    assert recovered_empty.get_stats()["total_feedbacks"] == 0


def test_validation_and_unknown_state_handling(tmp_path: Path):
    agent = ContextualBanditAgent(policy_path=tmp_path / "val.json")

    # Estado desconocido no vacío se inicializa con seguridad
    sel = agent.select_action("Potato___Custom_state")
    assert sel["strategy"] in ACTIONS

    with pytest.raises(ValueError):
        agent.select_action("")
    with pytest.raises(ValueError):
        agent.select_action(None)  # type: ignore[arg-type]

    for bad_rating in (0, 6, -1, True, False, "5", 4.5):
        with pytest.raises(ValueError):
            agent.update("Potato___healthy", "tecnico", bad_rating)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        agent.update("Potato___healthy", "invalida", 5)
