"""Najprostszy prototyp: agent idzie prosto do celu (bez interakcji)."""
from .base import Model
from . import register


@register("simple")
class SimpleGoalModel(Model):
    name = "simple"

    def step(self, agents, env, dt):
        for ag in agents:
            ag.vel = ag.desired_speed * ag.direction_to_goal()
            ag.pos = ag.pos + ag.vel * dt
            env.resolve_wall_penetration(ag)
