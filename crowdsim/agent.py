"""Wspólna reprezentacja agenta dla WSZYSTKICH modeli."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Agent:
    id: int
    pos: np.ndarray
    goal: np.ndarray
    radius: float = 0.1
    desired_speed: float = 1.34
    max_speed: float = 2.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    extra: dict = field(default_factory=dict)
    spawn_time: float = 0.0
    exit_time: float | None = None

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)
        self.vel = np.asarray(self.vel, dtype=float)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def direction_to_goal(self) -> np.ndarray:
        d = self.goal - self.pos
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.zeros(2)
