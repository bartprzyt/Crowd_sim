"""Środowisko: granice, przeszkody, wejścia (spawn) i wyjścia."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def closest_point_on_segment(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return a.copy()
    t = np.clip(float((p - a) @ ab) / denom, 0.0, 1.0)
    return a + t * ab


@dataclass
class Exit:
    a: np.ndarray
    b: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.a + self.b) / 2

    def distance(self, p) -> float:
        return float(np.linalg.norm(p - closest_point_on_segment(p, self.a, self.b)))


@dataclass
class SpawnZone:
    x: float
    y: float
    w: float
    h: float
    n: int
    goal_exit: int | None = None   # indeks wyjścia; None = najbliższe


class Environment:
    def __init__(self, width, height, walls=None, obstacles=None,
                 exits=None, spawn_zones=None):
        self.width, self.height = float(width), float(height)
        self.walls = []        # lista segmentów (a, b)
        self.obstacles = []    # opis do rysowania: dict
        self.exits: list[Exit] = []
        self.spawn_zones: list[SpawnZone] = []
        for a, b in (walls or []):
            self.add_wall(a, b)
        for o in (obstacles or []):
            self.add_obstacle(o)
        for a, b in (exits or []):
            self.exits.append(Exit(np.array(a, float), np.array(b, float)))
        self.spawn_zones = spawn_zones or []

    # --- budowa ---
    def add_wall(self, a, b):
        self.walls.append((np.array(a, float), np.array(b, float)))

    def add_obstacle(self, o: dict):
        """Przeszkoda prostokątna {'x','y','w','h'} -> 4 ściany."""
        self.obstacles.append(o)
        x, y, w, h = o["x"], o["y"], o["w"], o["h"]
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        for i in range(4):
            self.add_wall(pts[i], pts[(i + 1) % 4])

    # --- geometria ---
    def nearest_wall_point(self, p):
        """Zwraca (najbliższy punkt ściany, odległość)."""
        best, best_d = None, float("inf")
        for a, b in self.walls:
            q = closest_point_on_segment(p, a, b)
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best, best_d = q, d
        return best, best_d

    def nearest_exit(self, p) -> int:
        return int(np.argmin([e.distance(p) for e in self.exits]))

    def resolve_wall_penetration(self, agent):
        """Wypycha agenta ze ścian (twarde ograniczenie geometryczne)."""
        for a, b in self.walls:
            q = closest_point_on_segment(agent.pos, a, b)
            d = agent.pos - q
            dist = float(np.linalg.norm(d))
            if dist < agent.radius:
                n = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                agent.pos = q + n * agent.radius

    def has_reached_exit(self, agent, tol=0.3) -> bool:
        return any(e.distance(agent.pos) < tol for e in self.exits)
