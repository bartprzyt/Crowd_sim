from __future__ import annotations
import heapq
from collections import defaultdict
import numpy as np

from .base import Model
from . import register


@register("cellular")
@register("ca")
class CellularAutomatonModel(Model):
    """Model automatu komórkowego z polem statycznym i dynamicznym."""

    name = "cellular"

    def __init__(
        self,
        cell_size: float = 0.4,
        ks: float = 3.0,
        kd: float = 0.0,
        decay: float = 0.1,
        diffusion: float = 0.05,
        friction: float = 0.0,
        **params,
    ):
        """Inicjalizacja parametrów automatu komórkowego.

        :param cell_size: Rozmiar boku komórki w metrach.
        :param ks: Współczynnik czułości na pole statyczne.
        :param kd: Współczynnik czułości na pole dynamiczne (ślady pieszych).
        :param decay: Współczynnik zaniku pola dynamicznego.
        :param diffusion: Współczynnik dyfuzji pola dynamicznego.
        :param friction: Prawdopodobieństwo zablokowania przy konflikcie o komórkę.
        """
        super().__init__(**params)
        self.cell_size = float(cell_size)
        self.ks = float(ks)
        self.kd = float(kd)
        self.decay = float(decay)
        self.diffusion = float(diffusion)
        self.friction = float(friction)

        self.rng = np.random.default_rng(params.get("seed", 0))
        self.cols = 0
        self.rows = 0
        self.static_field = np.empty((0, 0), dtype=float)
        self.dynamic_field = np.empty((0, 0), dtype=float)
        self.cell_map: dict[tuple[int, int], int] = {}
        self.current_cells: dict[int, tuple[int, int]] = {}
        self.target_cells: dict[int, tuple[int, int]] = {}

    def _cell_center(self, r: int, c: int) -> np.ndarray:
        """Zwraca współrzędne ciągłe środka komórki (r, c)."""
        return np.array([(c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size], dtype=float)

    def _pos_to_cell(self, pos: np.ndarray) -> tuple[int, int]:
        """Przelicza pozycję ciągłą na indeksy komórki (r, c)."""
        c = int(np.clip(pos[0] // self.cell_size, 0, self.cols - 1))
        r = int(np.clip(pos[1] // self.cell_size, 0, self.rows - 1))
        return r, c

    @staticmethod
    def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        """Sprawdza orientację punktów w 2D."""
        return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))

    @classmethod
    def _segment_intersects(cls, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
        """Sprawdza, czy odcinek p1-p2 przecina się z odcinkiem p3-p4."""
        return cls._ccw(p1, p3, p4) != cls._ccw(p2, p3, p4) and cls._ccw(p1, p2, p3) != cls._ccw(p1, p2, p4)

    def _is_transition_blocked(self, p1: np.ndarray, p2: np.ndarray, walls: list) -> bool:
        """Sprawdza, czy ruch z p1 do p2 przecina którąkolwiek ze ścian."""
        for w_a, w_b in walls:
            if self._segment_intersects(p1, p2, w_a, w_b):
                return True
        return False

    def _build_static_field(self, env, clearance: float) -> np.ndarray:
        """Wyznacza pole statyczne odległości do wyjść algorytmem Dijkstry."""
        field = np.full((self.rows, self.cols), np.inf, dtype=float)
        is_wall = np.zeros((self.rows, self.cols), dtype=bool)
        pq: list[tuple[float, int, int]] = []

        for r in range(self.rows):
            for c in range(self.cols):
                p = self._cell_center(r, c)
                if any(e.distance(p) <= self.cell_size * 0.8 for e in env.exits):
                    field[r, c] = 0.0
                    heapq.heappush(pq, (0.0, r, c))
                else:
                    _, d = env.nearest_wall_point(p)
                    if d < clearance:
                        is_wall[r, c] = True

        while pq:
            d, r, c = heapq.heappop(pq)
            if d > field[r, c]:
                continue
            p1 = self._cell_center(r, c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if is_wall[nr, nc]:
                            continue
                        p2 = self._cell_center(nr, nc)
                        if self._is_transition_blocked(p1, p2, env.walls):
                            continue
                        step_dist = float(np.hypot(dr * self.cell_size, dc * self.cell_size))
                        new_d = d + step_dist
                        if new_d < field[nr, nc]:
                            field[nr, nc] = new_d
                            heapq.heappush(pq, (new_d, nr, nc))
        return field

    def _find_free_cell(self, start_r: int, start_c: int) -> tuple[int, int]:
        """Wyszukuje przeszukiwaniem wszerz najbliższą wolną i osiągalną komórkę."""
        if (start_r, start_c) not in self.cell_map and np.isfinite(self.static_field[start_r, start_c]):
            return start_r, start_c

        queue = [(start_r, start_c)]
        visited = {(start_r, start_c)}
        while queue:
            curr_r, curr_c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    if (nr, nc) not in self.cell_map and np.isfinite(self.static_field[nr, nc]):
                        return nr, nc
                    queue.append((nr, nc))
        return start_r, start_c

    def reset(self, agents, env):
        """Inicjalizacja siatki, pól odległości i rozmieszczenie agentów na komórkach."""
        self.cols = int(np.ceil(env.width / self.cell_size))
        self.rows = int(np.ceil(env.height / self.cell_size))
        min_radius = min((a.radius for a in agents), default=0.2)
        clearance = min_radius * 0.95
        self.static_field = self._build_static_field(env, clearance)
        self.dynamic_field = np.zeros((self.rows, self.cols), dtype=float)
        self.cell_map.clear()
        self.current_cells.clear()
        self.target_cells.clear()

        for a in agents:
            r, c = self._pos_to_cell(a.pos)
            fr, fc = self._find_free_cell(r, c)
            self.cell_map[(fr, fc)] = a.id
            self.current_cells[a.id] = (fr, fc)
            self.target_cells[a.id] = (fr, fc)
            a.pos = self._cell_center(fr, fc)

    def _update_dynamic_field(self):
        """Aktualizuje pole dynamiczne: dyfuzja do sąsiadów i wykładniczy zanik."""
        if self.kd <= 0.0:
            return

        diffused = np.zeros_like(self.dynamic_field)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(self.dynamic_field, dr, axis=0), dc, axis=1)
                diffused += shifted
        self.dynamic_field = (
            (1.0 - self.diffusion) * self.dynamic_field + (self.diffusion / 8.0) * diffused
        ) * (1.0 - self.decay)

    def step(self, agents, env, dt: float) -> None:
        """Krok czasowy automatu: gradientowy wybór komórek, rozwiązanie konfliktów i ruch."""
        active_ids = {a.id for a in agents}
        for cell, aid in list(self.cell_map.items()):
            if aid not in active_ids:
                del self.cell_map[cell]
        self.current_cells = {aid: c for aid, c in self.current_cells.items() if aid in active_ids}
        self.target_cells = {aid: c for aid, c in self.target_cells.items() if aid in active_ids}

        for a in agents:
            if a.id not in self.current_cells:
                r, c = self._pos_to_cell(a.pos)
                fr, fc = self._find_free_cell(r, c)
                self.cell_map[(fr, fc)] = a.id
                self.current_cells[a.id] = (fr, fc)
                self.target_cells[a.id] = (fr, fc)
                a.pos = self._cell_center(fr, fc)

        self._update_dynamic_field()

        proposals: dict[tuple[int, int], list[tuple[float, object]]] = defaultdict(list)
        for a in agents:
            curr = self.current_cells[a.id]
            target = self.target_cells[a.id]
            if curr == target:
                if self.static_field[curr] > 0.0:
                    r, c = curr
                    p1 = self._cell_center(r, c)
                    best_cell = None
                    best_score = 0.0

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                if np.isfinite(self.static_field[nr, nc]) and (nr, nc) not in self.cell_map:
                                    p2 = self._cell_center(nr, nc)
                                    if not self._is_transition_blocked(p1, p2, env.walls):
                                        step_len = float(np.hypot(dr * self.cell_size, dc * self.cell_size))
                                        grad = (self.static_field[r, c] - self.static_field[nr, nc]) / step_len
                                        if grad > 0.0:
                                            d_val = self.dynamic_field[nr, nc] if self.kd > 0.0 else 0.0
                                            score = grad + self.kd * d_val
                                            if score > best_score:
                                                best_score = score
                                                best_cell = (nr, nc)

                    if best_cell is not None:
                        proposals[best_cell].append((best_score, a))

        for cand_cell, suitors in proposals.items():
            if len(suitors) > 1 and self.friction > 0.0 and self.rng.random() < self.friction:
                continue

            if len(suitors) == 1:
                winner = suitors[0][1]
            else:
                suitors.sort(key=lambda x: x[0], reverse=True)
                top_score = suitors[0][0]
                tied = [s[1] for s in suitors if s[0] >= top_score - 1e-6]
                winner = tied[0] if len(tied) == 1 else self.rng.choice(tied)

            self.cell_map[cand_cell] = winner.id
            self.target_cells[winner.id] = cand_cell

        for a in agents:
            curr = self.current_cells[a.id]
            target = self.target_cells[a.id]

            if self.static_field[curr] == 0.0 and curr == target:
                a.vel = a.desired_speed * a.direction_to_goal()
                a.pos = a.pos + a.vel * dt
                if self.kd > 0.0:
                    self.dynamic_field[curr] += 1.0
            elif curr != target:
                target_pos = self._cell_center(*target)
                diff = target_pos - a.pos
                dist = float(np.linalg.norm(diff))
                step_dist = a.desired_speed * dt

                if step_dist >= dist or dist < 1e-4:
                    a.pos = target_pos
                    if self.cell_map.get(curr) == a.id:
                        del self.cell_map[curr]
                    self.current_cells[a.id] = target
                    a.vel = np.zeros(2, dtype=float)
                    if self.kd > 0.0:
                        self.dynamic_field[target] += 1.0
                else:
                    a.vel = (diff / dist) * a.desired_speed
                    a.pos = a.pos + a.vel * dt
            else:
                a.vel = np.zeros(2, dtype=float)

            env.resolve_wall_penetration(a)
