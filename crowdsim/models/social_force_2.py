
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
 
    def __init__(self, env, exit_obj, cell=0.2, clearance=0.2):
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
        free = (dmin >= clearance).reshape(self.ny, self.nx)
 
        a, b = exit_obj.a, exit_obj.b
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        dexit = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1).reshape(self.ny, self.nx)
        src = dexit <= 0.75 * cell + 1e-9
        free = free | src
 
        self.dist = self._dijkstra(free, src)
        self.dir = self._direction_field(self.dist)
 
    def _dijkstra(self, free, src):
        ny, nx, h = self.ny, self.nx, self.h
        dist = np.full((ny, nx), np.inf)
        heap = []
        for j, i in zip(*np.nonzero(src)):
            dist[j, i] = 0.0
            heap.append((0.0, int(j), int(i)))
        heapq.heapify(heap)
        while heap:
            d, j, i = heapq.heappop(heap)
            if d > dist[j, i]:
                continue
            for dj, di in _OFFSETS:
                jj, ii = j + dj, i + di
                if not (0 <= jj < ny and 0 <= ii < nx) or not free[jj, ii]:
                    continue
                if dj != 0 and di != 0 and not (free[j + dj, i] and free[j, i + di]):
                    continue
                nd = d + (_SQRT2 if dj and di else 1.0) * h
                if nd < dist[jj, ii]:
                    dist[jj, ii] = nd
                    heapq.heappush(heap, (nd, jj, ii))
        return dist
 
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
                 repulsion_range=0.3,
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
        self.rng = np.random.default_rng(seed)
        self._fields = {}
 
    def reset(self, agents, env):
        self._fields = {}
        if not (self.use_navigation and getattr(env, "exits", None)):
            return
        for ag in agents:
            key = tuple(np.round(ag.goal, 3))
            if key in self._fields:
                continue
            ex = min(env.exits, key=lambda e: float(np.linalg.norm(e.center - ag.goal)))
            try:
                self._fields[key] = _NavField(env, ex, self.nav_cell, self.nav_clearance)
            except Exception:
                self._fields[key] = None
 
    def _desired_directions(self, agents, P):
        E = np.zeros_like(P)
        groups = {}
        for i, ag in enumerate(agents):
            groups.setdefault(tuple(np.round(ag.goal, 3)), []).append(i)
        for key, idx in groups.items():
            idx = np.array(idx)
            field = self._fields.get(key)
            d = field.direction(P[idx]) if field is not None else np.zeros((len(idx), 2))
            direct = np.array([agents[i].direction_to_goal() for i in idx])
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
        for a, b in env.walls:
            ab = b - a
            t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            q = a + t[:, None] * ab
            d = P - q
            dd = np.linalg.norm(d, axis=1)
            nn = d / np.maximum(dd, 1e-9)[:, None]
            nn[dd < 1e-9] = np.array([1.0, 0.0])
            tt = np.stack([-nn[:, 1], nn[:, 0]], axis=1)
            ovw = R - dd
            gw = np.maximum(ovw, 0.0)
            fw = self.Aw * np.exp(np.minimum(ovw / self.Bw, 30.0)) + self.k * gw
            vt = np.einsum("ik,ik->i", V, tt)
            acc += fw[:, None] * nn - (self.kappa * gw * vt)[:, None] * tt
        return acc
 
    def step(self, agents, env, dt: float) -> None:
        if not agents:
            return
        P = np.array([a.pos for a in agents], float)
        V = np.array([a.vel for a in agents], float)
        R = np.array([a.radius for a in agents], float)
        V0 = np.array([a.desired_speed for a in agents], float)
        VM = np.minimum(np.array([a.max_speed for a in agents], float), self.speed_factor * V0)
 
        nsub = max(1, int(math.ceil(dt / self.max_dt)))
        h = dt / nsub
        for _ in range(nsub):
            E = self._desired_directions(agents, P)
            acc = self._accelerations(P, V, R, V0, E, env)
            if self.noise > 0:
                acc += self.rng.normal(0.0, self.noise, acc.shape)
            V = V + acc * h
            sp = np.linalg.norm(V, axis=1)
            over = sp > VM
            V[over] *= (VM[over] / sp[over])[:, None]
            P = P + V * h
            for i, ag in enumerate(agents):
                before = P[i].copy()
                ag.pos = P[i]
                env.resolve_wall_penetration(ag)
                if self.wall_velocity_fix:
                    delta = ag.pos - before
                    dn = float(np.linalg.norm(delta))
                    if dn > 1e-12:
                        nrm = delta / dn
                        vn = float(V[i] @ nrm)
                        if vn < 0.0:
                            V[i] = V[i] - vn * nrm
                P[i] = ag.pos
 
        for i, ag in enumerate(agents):
            ag.pos, ag.vel = P[i].copy(), V[i].copy()