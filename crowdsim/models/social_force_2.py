from __future__ import annotations

import heapq
import math

import numpy as np

from ..environment import closest_point_on_segment
from .base import Model
from . import register

_SQRT2 = math.sqrt(2.0)
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class _NavField:

    def __init__(self, env, exit_obj, cell=0.2, clearance=0.2, wall_pref=0.4, wall_weight=4.0):
        self.h = cell
        self.nx = max(1, int(math.ceil(env.width / cell)))
        self.ny = max(1, int(math.ceil(env.height / cell)))
        xs = (np.arange(self.nx) + 0.5) * cell
        ys = (np.arange(self.ny) + 0.5) * cell
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        dmin = np.full(len(pts), np.inf)
        for a, b in env.walls:
            ab = b - a
            t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            dmin = np.minimum(dmin, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
        dmin = dmin.reshape(self.ny, self.nx)
        free = dmin >= clearance - 1e-9          # tolerancja: unika losowości przy równości
        # koszt przejścia: >1 blisko ścian -> ścieżki trzymają się osi przejścia
        # i nie ścinają narożników (inaczej agent staje w lokalnym minimum przy rogu ławki)
        span = max(wall_pref - clearance, 1e-9)
        pen = np.clip((wall_pref - dmin) / span, 0.0, 1.0)
        self.cost = 1.0 + wall_weight * pen ** 2

        a, b = exit_obj.a, exit_obj.b
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        dexit = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1).reshape(self.ny, self.nx)
        src = dexit <= 0.75 * cell + 1e-9
        free = free | src

        self.free = free
        self.src = src
        self.base_cost = self.cost.copy()
        self._crowd_ema = np.zeros((self.ny, self.nx), dtype=float)
        self.dist = self._dijkstra(free, src, self.cost)
        self.dir = self._direction_field(self.dist)

    @staticmethod
    def _diffuse_neumann(grid, free, steps, rate=0.22):
        g = grid.copy()
        link_v = free[:-1, :] & free[1:, :]
        link_h = free[:, :-1] & free[:, 1:]
        for _ in range(steps):
            fv = rate * (g[:-1, :] - g[1:, :]) * link_v
            g[:-1, :] -= fv
            g[1:, :] += fv
            fh = rate * (g[:, :-1] - g[:, 1:]) * link_h
            g[:, :-1] -= fh
            g[:, 1:] += fh
        return g

    def update_congestion(self, P, V, V0, r_mean, weight=4.0, ema=0.45):
        occ = np.zeros((self.ny, self.nx), dtype=float)
        if len(P) > 0:
            sp = np.linalg.norm(V, axis=1)
            slow = 1.0 - np.clip(sp / np.maximum(V0, 1e-6), 0.0, 1.0)
            w = 0.35 + 0.65 * slow
            ix = np.clip((P[:, 0] / self.h).astype(int), 0, self.nx - 1)
            iy = np.clip((P[:, 1] / self.h).astype(int), 0, self.ny - 1)
            np.add.at(occ, (iy, ix), w)
        occ /= (self.h * self.h)
        steps = max(4, min(40, int(round((0.45 / self.h) ** 2 / 0.45))))
        dens = self._diffuse_neumann(occ, self.free, steps=steps, rate=0.22)
        self._crowd_ema = ema * dens + (1.0 - ema) * self._crowd_ema
        dens_ref = 1.0 / max((2.0 * r_mean) ** 2 * 2.5, 0.1)
        cong = np.clip(self._crowd_ema / dens_ref, 0.0, 2.5)
        cong[self.src] = 0.0
        self.cost = self.base_cost + weight * cong
        self.dist = self._dijkstra(self.free, self.src, self.cost)
        self.dir = self._direction_field(self.dist)

    def _dijkstra(self, free, src, cost):
        ny, nx, h = self.ny, self.nx, self.h
        dist = np.full((ny, nx), np.inf)
        free_l, cost_l = free.tolist(), cost.tolist()
        dl = [[math.inf] * nx for _ in range(ny)]
        heap = []
        for j, i in zip(*np.nonzero(src)):
            dl[j][i] = 0.0
            heap.append((0.0, int(j), int(i)))
        heapq.heapify(heap)
        while heap:
            d, j, i = heapq.heappop(heap)
            if d > dl[j][i]:
                continue
            cj = cost_l[j][i]
            for dj, di in _OFFSETS:
                jj, ii = j + dj, i + di
                if not (0 <= jj < ny and 0 <= ii < nx) or not free_l[jj][ii]:
                    continue
                if dj != 0 and di != 0 and not (free_l[j + dj][i] and free_l[j][i + di]):
                    continue
                nd = d + (_SQRT2 if dj and di else 1.0) * h * 0.5 * (cj + cost_l[jj][ii])
                if nd < dl[jj][ii]:
                    dl[jj][ii] = nd
                    heapq.heappush(heap, (nd, jj, ii))
        return np.array(dl)

    def _direction_field(self, dist):
        ny, nx = dist.shape
        pad = np.pad(dist, 1, constant_values=np.inf)
        vec = np.zeros((ny, nx, 2))
        for dj, di in _OFFSETS:
            nb = pad[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
            ok = np.isfinite(dist) & np.isfinite(nb)
            with np.errstate(invalid="ignore"):
                gain = np.where(ok, np.maximum(dist - nb, 0.0), 0.0)
            norm = math.hypot(dj, di)
            vec[..., 0] += gain * di / norm / norm
            vec[..., 1] += gain * dj / norm / norm
        n = np.linalg.norm(vec, axis=2)
        has = n > 1e-12
        vec[has] /= n[has][:, None]
        for _ in range(3):
            pv = np.pad(vec, ((1, 1), (1, 1), (0, 0)))
            ph = np.pad(has, 1)
            acc = np.zeros_like(vec)
            cnt = np.zeros((ny, nx))
            for dj, di in _OFFSETS:
                m = ph[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
                acc += pv[1 + dj:1 + dj + ny, 1 + di:1 + di + nx] * m[..., None]
                cnt += m
            fill = (~has) & (cnt > 0)
            vec[fill] = acc[fill] / cnt[fill][:, None]
            nn = np.linalg.norm(vec, axis=2)
            vec[fill] /= np.maximum(nn[fill], 1e-12)[:, None]
            has = has | fill
        return vec

    def unreachable(self, pos):
        """True dla pozycji, w pobliżu których (okno 5x5 komórek) nie ma żadnej komórki z dojściem do wyjścia."""
        fin = np.isfinite(self.dist)
        pad = np.pad(fin, 2, constant_values=False)
        win = np.zeros_like(fin)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                win |= pad[2 + dj:2 + dj + self.ny, 2 + di:2 + di + self.nx]
        i = np.clip((pos[:, 0] / self.h).astype(int), 0, self.nx - 1)
        j = np.clip((pos[:, 1] / self.h).astype(int), 0, self.ny - 1)
        return ~win[j, i]

    def direction(self, pos):
        fx = np.clip(pos[:, 0] / self.h - 0.5, 0, self.nx - 1 - 1e-9)
        fy = np.clip(pos[:, 1] / self.h - 0.5, 0, self.ny - 1 - 1e-9)
        i0, j0 = fx.astype(int), fy.astype(int)
        i1, j1 = np.minimum(i0 + 1, self.nx - 1), np.minimum(j0 + 1, self.ny - 1)
        wx, wy = (fx - i0)[:, None], (fy - j0)[:, None]
        d = (self.dir[j0, i0] * (1 - wx) * (1 - wy) + self.dir[j0, i1] * wx * (1 - wy)
             + self.dir[j1, i0] * (1 - wx) * wy + self.dir[j1, i1] * wx * wy)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return np.where(n > 1e-9, d / np.maximum(n, 1e-12), 0.0)


@register("social_force2")
class SocialForceModel(Model):
    name = "social_force2"

    def __init__(self,
                 relaxation_time=0.5,
                 repulsion_strength=2.1,
                 repulsion_range=0.2,
                 wall_strength=None,
                 wall_range=None,
                 body_force=1500.0,
                 friction=3000.0,
                 anisotropy=1.0,
                 speed_factor=1.3,
                 noise=0.0,
                 max_dt=0.01,
                 wall_velocity_fix=True,
                 use_navigation=True,
                 nav_cell=0.2, nav_clearance=0.1,
                 nav_wall_pref=0.4, nav_wall_weight=4.0,
                 congestion_weight=4.0, congestion_interval=0.5, congestion_ema=0.45,
                 seed=0, **kw):
        super().__init__(relaxation_time=relaxation_time, repulsion_strength=repulsion_strength,
                         repulsion_range=repulsion_range, **kw)
        self.tau = relaxation_time
        self.A, self.B = repulsion_strength, repulsion_range
        self.Aw = repulsion_strength if wall_strength is None else wall_strength
        self.Bw = repulsion_range if wall_range is None else wall_range
        self.k, self.kappa = body_force, friction
        self.lam, self.speed_factor, self.noise = anisotropy, speed_factor, noise
        self.max_dt = max_dt
        self.use_navigation = use_navigation
        self.wall_velocity_fix = wall_velocity_fix
        self.nav_cell, self.nav_clearance = nav_cell, nav_clearance
        self.nav_wall_pref, self.nav_wall_weight = nav_wall_pref, nav_wall_weight
        self.congestion_weight = float(congestion_weight)
        self.congestion_interval = float(congestion_interval)
        self.congestion_ema = float(congestion_ema)
        self.rng = np.random.default_rng(seed)
        self._fields = {}
        self._wall_env = None
        self._cong_timer = 0.0
        self._r_mean = 0.2
        self.nav_version = 0

    def reset(self, agents, env):
        self._fields = {}
        self._wall_env = None
        self._cong_timer = 0.0
        self._r_mean = float(np.mean([a.radius for a in agents])) if agents else 0.2
        self.nav_version += 1
        if not (self.use_navigation and getattr(env, "exits", None)) or not agents:
            return
        # komórka siatki nie może być grubsza niż pół promienia agenta - inaczej wąskie
        # przejścia (np. 0.4 m między ławkami) mogą w ogóle nie mieć wolnych komórek
        rmin = min(a.radius for a in agents)
        cell = min(self.nav_cell, 0.5 * rmin)
        if cell < 0.9 * self.nav_cell:
            print(f"[social_force2] nav_cell {self.nav_cell} -> {cell:.3f} (dopasowane do promienia agentów)")
        groups = {}
        for i, ag in enumerate(agents):
            groups.setdefault(tuple(np.round(ag.goal, 3)), []).append(i)
        for key, idx in groups.items():
            goal = agents[idx[0]].goal
            ex = min(env.exits, key=lambda e: float(np.linalg.norm(e.center - goal)))
            Pg = np.array([agents[i].pos for i in idx], float)
            field = None
            for factor in (1.0, 0.75, 0.5):
                try:
                    cand = _NavField(env, ex, cell, self.nav_clearance * factor,
                                     self.nav_wall_pref, self.nav_wall_weight)
                except Exception:
                    break
                field = cand
                if not cand.unreachable(Pg).any():
                    break
            if field is not None:
                n_bad = int(field.unreachable(Pg).sum())
                if n_bad:
                    print(f"[social_force2] UWAGA: {n_bad} agentów startuje w miejscu bez dojścia do wyjścia "
                          f"w siatce nawigacji (cel {key}); zmniejsz nav_cell lub nav_clearance")
            self._fields[key] = field

    @staticmethod
    def _goal_groups(agents):
        groups = {}
        for i, ag in enumerate(agents):
            groups.setdefault(tuple(np.round(ag.goal, 3)), []).append(i)
        return {k: np.array(v) for k, v in groups.items()}

    def _desired_directions(self, groups, G, P):
        """G - macierz celów (N,2); nie czyta ag.pos, tylko aktualne P."""
        E = np.zeros_like(P)
        for key, idx in groups.items():
            field = self._fields.get(key)
            d = field.direction(P[idx]) if field is not None else np.zeros((len(idx), 2))
            direct = G[idx] - P[idx]
            direct /= np.maximum(np.linalg.norm(direct, axis=1, keepdims=True), 1e-9)
            bad = np.linalg.norm(d, axis=1) < 1e-9
            d[bad] = direct[bad]
            E[idx] = d
        return E

    # --- siły ---
    def _accelerations(self, P, V, R, V0, E, env):
        acc = (V0[:, None] * E - V) / self.tau

        # agent - agent
        diff = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        safe = np.maximum(dist, 1e-9)
        nrm = diff / safe[..., None]
        tang = np.stack([-nrm[..., 1], nrm[..., 0]], axis=2)
        rij = R[:, None] + R[None, :]
        ov = rij - dist
        fsoc = self.A * np.exp(np.minimum(ov / self.B, 30.0))
        if self.lam < 1.0:
            sp = np.linalg.norm(V, axis=1, keepdims=True)
            ev = np.where(sp > 1e-6, V / np.maximum(sp, 1e-9), E)
            cosphi = -np.einsum("ik,ijk->ij", ev, nrm)
            fsoc = fsoc * (self.lam + (1 - self.lam) * (1 + cosphi) / 2)
        g = np.maximum(ov, 0.0)
        dvt = np.einsum("ijk,ijk->ij", V[None, :, :] - V[:, None, :], tang)
        ft = self.kappa * g * dvt
        acc += ((fsoc + self.k * g)[..., None] * nrm).sum(axis=1)
        acc += (ft[..., None] * tang).sum(axis=1)

        # agent - ściany / przeszkody
        # Jedna siła od najbliższego punktu KAŻDEJ przeszkody (a nie od każdego odcinka osobno):
        # sumowanie odcinków dodaje siły od narożników i tworzy barierę we wlotach wąskich przejść.
        D, dd = self._wall_groups_nearest(P, env)          # (G,N,2), (G,N)
        nn = D / np.maximum(dd, 1e-9)[..., None]
        nn = np.where((dd < 1e-9)[..., None], np.array([1.0, 0.0]), nn)
        tt = np.stack([-nn[..., 1], nn[..., 0]], axis=2)
        ovw = R[None, :] - dd
        gw = np.maximum(ovw, 0.0)
        fw = self.Aw * np.exp(np.minimum(ovw / self.Bw, 30.0)) + self.k * gw
        vt = np.einsum("nk,gnk->gn", V, tt)
        acc += (fw[..., None] * nn - (self.kappa * gw * vt)[..., None] * tt).sum(axis=0)
        return acc

    def _cache_walls(self, env):
        W = env.walls
        self._wA = np.array([a for a, _ in W], float)
        self._wB = np.array([b for _, b in W], float)
        n_obs = len(getattr(env, "obstacles", []))
        n_out = len(W) - 4 * n_obs          # Environment: najpierw ściany, potem 4 odcinki na przeszkodę
        if n_out < 0:
            n_out, n_obs = len(W), 0
        self._n_out, self._n_obs = n_out, n_obs
        self._wall_env = env

    def _wall_groups_nearest(self, P, env):
        """Dla każdej ściany zewnętrznej i każdej przeszkody: wektor od najbliższego punktu do agenta."""
        if getattr(self, "_wall_env", None) is not env:
            self._cache_walls(env)
        A, B = self._wA, self._wB
        AB = B - A
        den = np.maximum((AB * AB).sum(1), 1e-12)
        t = np.clip(((P[None] - A[:, None]) * AB[:, None]).sum(2) / den[:, None], 0.0, 1.0)
        D = P[None] - (A[:, None] + t[..., None] * AB[:, None])      # (W,N,2)
        dd = np.linalg.norm(D, axis=2)                                # (W,N)
        n_out, n_obs = self._n_out, self._n_obs
        if n_obs:
            N = P.shape[0]
            Db = D[n_out:].reshape(n_obs, 4, N, 2)
            ddb = dd[n_out:].reshape(n_obs, 4, N)
            arg = ddb.argmin(axis=1)                                  # (n_obs,N)
            Dsel = np.take_along_axis(Db, arg[:, None, :, None], axis=1)[:, 0]
            ddsel = np.take_along_axis(ddb, arg[:, None, :], axis=1)[:, 0]
            D = np.concatenate([D[:n_out], Dsel], axis=0)
            dd = np.concatenate([dd[:n_out], ddsel], axis=0)
        return D, dd

    def _resolve_walls(self, P, R, env):
        """Wypycha wszystkich agentów ze ścian naraz (pętla tylko po ścianach)."""
        for a, b in env.walls:
            ab = b - a
            t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            q = a + t[:, None] * ab
            d = P - q
            dd = np.linalg.norm(d, axis=1)
            hit = dd < R
            if not hit.any():
                continue
            n = np.where((dd > 1e-9)[:, None], d / np.maximum(dd, 1e-9)[:, None],
                         np.array([1.0, 0.0]))
            P[hit] = q[hit] + n[hit] * R[hit][:, None]
        return P

    def step(self, agents, env, dt: float) -> None:
        if not agents:
            return
        P = np.array([a.pos for a in agents], float)
        V = np.array([a.vel for a in agents], float)
        R = np.array([a.radius for a in agents], float)
        V0 = np.array([a.desired_speed for a in agents], float)
        G = np.array([a.goal for a in agents], float)
        VM = np.minimum(np.array([a.max_speed for a in agents], float), self.speed_factor * V0)
        groups = self._goal_groups(agents)

        nsub = max(1, int(math.ceil(dt / self.max_dt)))
        h = dt / nsub
        for _ in range(nsub):
            E = self._desired_directions(groups, G, P)
            acc = self._accelerations(P, V, R, V0, E, env)
            if self.noise > 0:
                acc += self.rng.normal(0.0, self.noise, acc.shape)
            V = V + acc * h
            sp = np.linalg.norm(V, axis=1)
            over = sp > VM
            V[over] *= (VM[over] / sp[over])[:, None]
            P = P + V * h

            # twarde wypychanie ze ścian (wektorowo) + usunięcie składowej prędkości w ścianę
            before = P.copy()
            P = self._resolve_walls(P, R, env)
            if self.wall_velocity_fix:
                delta = P - before
                dn = np.linalg.norm(delta, axis=1)
                moved = dn > 1e-12
                nrm = delta / np.maximum(dn, 1e-12)[:, None]
                vn = np.einsum("ik,ik->i", V, nrm)
                cut = moved & (vn < 0.0)
                V[cut] -= vn[cut][:, None] * nrm[cut]

        for i, ag in enumerate(agents):
            ag.pos, ag.vel = P[i].copy(), V[i].copy()

        if self.use_navigation and self.congestion_weight > 0.0 and self._fields:
            self._cong_timer += dt
            if self._cong_timer >= self.congestion_interval:
                self._cong_timer = 0.0
                for field in self._fields.values():
                    if field is not None:
                        field.update_congestion(
                            P, V, V0, self._r_mean,
                            weight=self.congestion_weight,
                            ema=self.congestion_ema,
                        )
                self.nav_version += 1