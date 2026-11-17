import json
import os
import random
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

STATES: tuple[str, ...] = (
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
)
ACTIONS: tuple[str, ...] = ("tecnico", "practico", "ecologico")

DEFAULT_EPSILON: float = 0.2
DEFAULT_ALPHA: float = 0.2
MAX_RECENT_FEEDBACK: int = 50


class ContextualBanditAgent:
    """
    Agente de Aprendizaje por Refuerzo (Contextual Multi-Armed Bandit) con
    política ε-greedy y tabla Q(s, a) persistida atómicamente en JSON UTF-8.
    """

    STATES = STATES
    ACTIONS = ACTIONS

    def __init__(
        self,
        policy_path: Path | str | None = None,
        epsilon: float = DEFAULT_EPSILON,
        alpha: float = DEFAULT_ALPHA,
        seed: int | None = None,
    ) -> None:
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError("epsilon debe ser un número en [0.0, 1.0].")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha debe ser un número en (0.0, 1.0].")

        base_dir = Path(__file__).resolve().parent
        self.policy_path: Path = Path(policy_path) if policy_path is not None else (base_dir / "data" / "rl_policy.json")
        self.epsilon: float = float(epsilon)
        self.alpha: float = float(alpha)

        self._lock = threading.Lock()
        self._rng = random.Random(seed)

        self.q_table: dict[str, dict[str, float]] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.action_counts: dict[str, int] = {}
        self.total_feedbacks: int = 0
        self.total_reward: float = 0.0
        self.recent_feedback: list[dict[str, Any]] = []
        self.updated_at: Optional[str] = None

        self._init_defaults_locked()
        self._load_locked()

    def _init_defaults_locked(self) -> None:
        self.q_table = {s: {a: 0.0 for a in ACTIONS} for s in STATES}
        self.counts = {s: {a: 0 for a in ACTIONS} for s in STATES}
        self.action_counts = {a: 0 for a in ACTIONS}
        self.total_feedbacks = 0
        self.total_reward = 0.0
        self.recent_feedback = []
        self.updated_at = None

    def _ensure_state_locked(self, state: str) -> None:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ACTIONS}
        else:
            for a in ACTIONS:
                self.q_table[state].setdefault(a, 0.0)

        if state not in self.counts:
            self.counts[state] = {a: 0 for a in ACTIONS}
        else:
            for a in ACTIONS:
                self.counts[state].setdefault(a, 0)

    def _load_locked(self) -> None:
        if not self.policy_path.exists():
            return
        try:
            raw_text = self.policy_path.read_text(encoding="utf-8").strip()
            if not raw_text:
                return
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                return

            loaded_q = data.get("q_table")
            if isinstance(loaded_q, dict):
                for state, row in loaded_q.items():
                    if isinstance(state, str) and state.strip() and isinstance(row, dict):
                        self._ensure_state_locked(state)
                        for action in ACTIONS:
                            val = row.get(action, 0.0)
                            if isinstance(val, (int, float)) and not isinstance(val, bool):
                                self.q_table[state][action] = float(val)

            loaded_counts = data.get("counts")
            if isinstance(loaded_counts, dict):
                for state, row in loaded_counts.items():
                    if isinstance(state, str) and state.strip() and isinstance(row, dict):
                        self._ensure_state_locked(state)
                        for action in ACTIONS:
                            cnt = row.get(action, 0)
                            if isinstance(cnt, int) and not isinstance(cnt, bool) and cnt >= 0:
                                self.counts[state][action] = cnt

            loaded_action_counts = data.get("action_counts")
            if isinstance(loaded_action_counts, dict):
                for action in ACTIONS:
                    cnt = loaded_action_counts.get(action, 0)
                    if isinstance(cnt, int) and not isinstance(cnt, bool) and cnt >= 0:
                        self.action_counts[action] = cnt
            else:
                for action in ACTIONS:
                    self.action_counts[action] = sum(self.counts[s][action] for s in self.counts)

            tf = data.get("total_feedbacks", 0)
            if isinstance(tf, int) and not isinstance(tf, bool) and tf >= 0:
                self.total_feedbacks = tf

            tr = data.get("total_reward", 0.0)
            if isinstance(tr, (int, float)) and not isinstance(tr, bool):
                self.total_reward = float(tr)

            rf = data.get("recent_feedback", [])
            if isinstance(rf, list):
                self.recent_feedback = [item for item in rf if isinstance(item, dict)][-MAX_RECENT_FEEDBACK:]

            ua = data.get("updated_at")
            if isinstance(ua, str) or ua is None:
                self.updated_at = ua
        except Exception:
            # Recuperación limpia ante archivos vacíos o JSON corrupto
            self._init_defaults_locked()

    def _save_locked(self) -> None:
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        average_reward = round(self.total_reward / self.total_feedbacks, 6) if self.total_feedbacks > 0 else 0.0
        payload = {
            "states": list(self.q_table.keys()),
            "actions": list(ACTIONS),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "q_table": {s: {a: round(float(v), 6) for a, v in row.items()} for s, row in self.q_table.items()},
            "counts": {s: {a: int(c) for a, c in row.items()} for s, row in self.counts.items()},
            "action_counts": {a: int(self.action_counts[a]) for a in ACTIONS},
            "total_feedbacks": self.total_feedbacks,
            "total_reward": round(self.total_reward, 6),
            "average_reward": average_reward,
            "recent_feedback": list(self.recent_feedback),
            "updated_at": self.updated_at,
        }

        fd, tmp_name = tempfile.mkstemp(
            prefix=".rl_policy_",
            suffix=".tmp",
            dir=str(self.policy_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, self.policy_path)
        except Exception:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _validate_state(state: Any) -> str:
        if not isinstance(state, str) or not state.strip():
            raise ValueError("El estado (disease_id) debe ser una cadena no vacía.")
        return state.strip()

    @staticmethod
    def _validate_action(action: Any) -> str:
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError(f"Estrategia inválida: {action!r}. Valores permitidos: {ACTIONS}.")
        return action

    @staticmethod
    def _validate_rating(rating: Any) -> int:
        if isinstance(rating, bool) or not isinstance(rating, int) or not (1 <= rating <= 5):
            raise ValueError("La calificación debe ser un entero entre 1 y 5.")
        return rating

    def select_action(self, state: str) -> dict[str, Any]:
        """Selecciona una estrategia usando la política ε-greedy para el estado dado."""
        clean_state = self._validate_state(state)

        with self._lock:
            self._ensure_state_locked(clean_state)
            q_row = self.q_table[clean_state]

            if self._rng.random() < self.epsilon:
                is_exploration = True
                mode = "exploration"
                mode_label = "Exploración"
                strategy = self._rng.choice(ACTIONS)
            else:
                is_exploration = False
                mode = "exploitation"
                mode_label = "Explotación"
                max_q = max(q_row.values())
                best_actions = [a for a in ACTIONS if abs(q_row[a] - max_q) < 1e-12]
                strategy = self._rng.choice(best_actions)

            return {
                "state": clean_state,
                "strategy": strategy,
                "mode": mode,
                "mode_label": mode_label,
                "is_exploration": is_exploration,
                "epsilon": self.epsilon,
                "q_values": {a: round(float(q_row[a]), 6) for a in ACTIONS},
            }

    def update(
        self,
        state: str,
        action: str,
        rating: int,
        comment: Optional[str] = None,
    ) -> dict[str, Any]:
        """Actualiza Q(s, a) con recompensa normalizada r = (rating - 3) / 2.0 y persiste."""
        clean_state = self._validate_state(state)
        clean_action = self._validate_action(action)
        clean_rating = self._validate_rating(rating)
        if comment is not None and not isinstance(comment, str):
            raise ValueError("El comentario debe ser una cadena de texto.")

        reward = (clean_rating - 3) / 2.0

        with self._lock:
            self._ensure_state_locked(clean_state)
            q_before = float(self.q_table[clean_state][clean_action])
            q_after = q_before + self.alpha * (reward - q_before)

            self.q_table[clean_state][clean_action] = q_after
            self.counts[clean_state][clean_action] += 1
            self.action_counts[clean_action] += 1
            self.total_feedbacks += 1
            self.total_reward += reward

            avg_reward = self.total_reward / self.total_feedbacks
            timestamp = datetime.now(timezone.utc).isoformat()
            self.updated_at = timestamp

            clean_comment = (comment or "").strip()[:500]
            self.recent_feedback.append(
                {
                    "timestamp": timestamp,
                    "state": clean_state,
                    "action": clean_action,
                    "rating": clean_rating,
                    "reward": round(reward, 4),
                    "comment": clean_comment,
                    "q_before": round(q_before, 6),
                    "q_after": round(q_after, 6),
                }
            )
            if len(self.recent_feedback) > MAX_RECENT_FEEDBACK:
                self.recent_feedback = self.recent_feedback[-MAX_RECENT_FEEDBACK:]

            self._save_locked()

            return {
                "state": clean_state,
                "action": clean_action,
                "rating": clean_rating,
                "reward": round(reward, 4),
                "q_before": round(q_before, 6),
                "q_after": round(q_after, 6),
                "total_feedbacks": self.total_feedbacks,
                "average_reward": round(avg_reward, 6),
            }

    def get_stats(self) -> dict[str, Any]:
        """Devuelve métricas globales, tabla Q(s, a), conteos y feedback reciente."""
        with self._lock:
            average_reward = round(self.total_reward / self.total_feedbacks, 6) if self.total_feedbacks > 0 else 0.0
            best_actions = {
                s: max(ACTIONS, key=lambda a: row[a])
                for s, row in self.q_table.items()
            }
            return {
                "states": list(self.q_table.keys()),
                "actions": list(ACTIONS),
                "q_table": {s: {a: round(float(v), 6) for a, v in row.items()} for s, row in self.q_table.items()},
                "counts": {s: {a: int(c) for a, c in row.items()} for s, row in self.counts.items()},
                "action_counts": {a: int(self.action_counts[a]) for a in ACTIONS},
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "total_feedbacks": self.total_feedbacks,
                "total_reward": round(self.total_reward, 6),
                "average_reward": average_reward,
                "best_actions": best_actions,
                "recent_feedback": list(self.recent_feedback),
                "updated_at": self.updated_at,
            }
