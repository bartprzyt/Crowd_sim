"""Wczytywanie scenariuszy z JSON."""
import json
import numpy as np
from .agent import Agent
from .environment import Environment, SpawnZone


def load_scenario(path, seed=0):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    env = Environment(
        cfg["width"], cfg["height"],
        walls=cfg.get("walls", []),
        obstacles=cfg.get("obstacles", []),
        exits=cfg.get("exits", []),
        spawn_zones=[SpawnZone(**z) for z in cfg.get("spawn_zones", [])],
    )
    rng = np.random.default_rng(seed)
    agents, aid = [], 0
    r = cfg.get("agent_radius", 0.25)
    for z in env.spawn_zones:
        placed, tries = 0, 0
        while placed < z.n and tries < z.n * 500:
            tries += 1
            p = np.array([rng.uniform(z.x + r, z.x + z.w - r),
                          rng.uniform(z.y + r, z.y + z.h - r)])
            if any(np.linalg.norm(p - a.pos) < 2 * r for a in agents):
                continue
            if env.nearest_wall_point(p)[1] < r:
                continue
            ex = z.goal_exit if z.goal_exit is not None else env.nearest_exit(p)
            speed = abs(float(rng.normal(1.34, 0.26)))
            agents.append(Agent(id=aid, pos=p, goal=env.exits[ex].center.copy(),
                                radius=r, desired_speed=max(0.5, speed)))
            aid += 1
            placed += 1
    return env, agents
