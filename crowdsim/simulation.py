"""Silnik symulacji: pętla czasowa, wymiana modeli, logowanie."""
from __future__ import annotations
import numpy as np


class Simulation:
    def __init__(self, env, agents, model, dt=0.05, logger=None, seed=0):
        self.env, self.model, self.dt, self.logger = env, model, dt, logger
        self.agents = list(agents)
        self.evacuated = []
        self.t = 0.0
        self.rng = np.random.default_rng(seed)
        self.model.reset(self.agents, self.env)

    def set_model(self, model):
        """Przełączenie algorytmu w locie."""
        self.model = model
        self.model.reset(self.agents, self.env)

    @property
    def finished(self):
        return len(self.agents) == 0

    def step(self):
        self.model.step(self.agents, self.env, self.dt)
        self.t += self.dt
        still = []
        for a in self.agents:
            if self.env.has_reached_exit(a):
                a.exit_time = self.t
                self.evacuated.append(a)
            else:
                still.append(a)
        self.agents = still
        if self.logger:
            self.logger.log(self.t, self.agents)

    def run(self, t_max=300.0):
        while not self.finished and self.t < t_max:
            self.step()
        return self.summary()

    def summary(self):
        times = [a.exit_time for a in self.evacuated]
        return {
            "t": round(self.t, 2),
            "evacuated": len(self.evacuated),
            "remaining": len(self.agents),
            "evac_time_total": max(times) if times else None,
            "evac_time_mean": float(np.mean(times)) if times else None,
        }
