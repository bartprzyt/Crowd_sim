from __future__ import annotations

import numpy as np

from ..environment import closest_point_on_segment
from .base import Model
from . import register


@register("social_force")
class SocialForceModel(Model):
    name = "social_force"

    relaxation_time = 0.5
    repulsion_strength = 2.1
    repulsion_range = 0.3

    def step(self, agents, env, dt: float) -> None:
        positions = [agent.pos.copy() for agent in agents]
        velocities = [agent.vel.copy() for agent in agents]
        new_velocities = []

        for i, agent in enumerate(agents):
            desired_velocity = agent.desired_speed * agent.direction_to_goal()
            force = (desired_velocity - velocities[i]) / self.relaxation_time

            for j, other in enumerate(agents):
                if i == j:
                    continue
                difference = positions[i] - positions[j]
                distance = float(np.linalg.norm(difference))
                if distance < 1e-9:
                    direction = np.array([1.0, 0.0]) if agent.id < other.id else np.array([-1.0, 0.0])
                else:
                    direction = difference / distance
                overlap = agent.radius + other.radius - distance
                force += (
                    self.repulsion_strength
                    * np.exp(overlap / self.repulsion_range)
                    * direction
                )

            for wall_start, wall_end in env.walls:
                closest = closest_point_on_segment(positions[i], wall_start, wall_end)
                difference = positions[i] - closest
                distance = float(np.linalg.norm(difference))
                if distance < 1e-9:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = difference / distance
                force += (
                    self.repulsion_strength
                    * np.exp((agent.radius - distance) / self.repulsion_range)
                    * direction
                )

            velocity = velocities[i] + force * dt
            speed = float(np.linalg.norm(velocity))
            if speed > agent.max_speed:
                velocity = velocity * (agent.max_speed / speed)
            new_velocities.append(velocity)

        for agent, velocity in zip(agents, new_velocities):
            agent.vel = velocity
            agent.pos = agent.pos + agent.vel * dt
            env.resolve_wall_penetration(agent)
