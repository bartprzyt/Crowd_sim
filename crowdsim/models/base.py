"""WSPÓLNY INTERFEJS MODELU - każdy z czterech modeli dziedziczy po Model."""
from __future__ import annotations
from abc import ABC, abstractmethod


class Model(ABC):
    name = "base"

    def __init__(self, **params):
        self.params = params

    def reset(self, agents, env):
        """Opcjonalna inicjalizacja (np. siatka CA, floor field, ORCA)."""

    @abstractmethod
    def step(self, agents, env, dt: float) -> None:
        """Jeden krok czasowy: zaktualizuj agent.pos i agent.vel
        dla wszystkich agentów (w miejscu)."""
