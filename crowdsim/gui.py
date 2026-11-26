"""Graficzny interfejs użytkownika."""
from __future__ import annotations

from datetime import datetime
import inspect
import json
import math
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pygame

from .scenario import load_scenario
from .simulation import Simulation
from .logger import TrajectoryLogger
from .models import create_model, MODEL_REGISTRY


COLOR_MODES = ("speed", "density", "solid")
COLOR_MODE_LABELS = {
    "speed": "Kolor: Prędkość",
    "density": "Kolor: Zatłoczenie",
    "solid": "Kolor: Jednolity",
}
COLOR_LEGEND_LABELS = {
    "speed": ("swobodnie", "spowolnienie", "zator"),
    "density": ("luźno", "umiarkowanie", "ścisk"),
}

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _read_scenario_json(scenario_path: str) -> dict:
    try:
        with open(scenario_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_model_params(scenario_path: str, model_name: str) -> dict:
    cfg = _read_scenario_json(scenario_path)
    return dict(cfg.get("model_params", {}).get(model_name, {}))


def _get_default_model_params(model_name: str) -> dict:
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        return {}
    sig = inspect.signature(cls.__init__)
    defaults = {}
    for pname, param in sig.parameters.items():
        if pname in ("self", "seed") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[pname] = param.default
    return defaults


def _max_scale_for(
    width_m: float,
    height_m: float,
    screen_w: int,
    screen_h: int,
    sidebar_w: int = 310,
) -> int:
    avail_w = max(360, screen_w - sidebar_w - 80)
    avail_h = max(480, screen_h - 110)
    sx = avail_w / max(width_m, 1.0)
    sy = avail_h / max(height_m, 1.0)
    return max(12, int(min(sx, sy)))


def _make_csv_path(scenario_path: str, model_name: str) -> str:
    stem = Path(scenario_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return (out_dir / f"traj_{stem}_{model_name}_{ts}.csv").as_posix()


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * t)),
        int(round(c1[1] + (c2[1] - c1[1]) * t)),
        int(round(c1[2] + (c2[2] - c1[2]) * t)),
    )


def _ratio_to_color(ratio: float) -> tuple[int, int, int]:
    """Mapuje współczynnik z [0, 1] na gradient: zielony -> żółty -> czerwony."""
    t = float(np.clip(ratio, 0.0, 1.0))
    green = (35, 175, 105)
    yellow = (245, 195, 35)
    red = (225, 45, 45)
    if t <= 0.5:
        return _lerp_color(green, yellow, t * 2.0)
    return _lerp_color(yellow, red, (t - 0.5) * 2.0)


def _compute_density_ratios(agents) -> dict[int, float]:
    n = len(agents)
    if n == 0:
        return {}
    if n == 1:
        return {agents[0].id: 0.0}

    pos = np.array([a.pos for a in agents], dtype=float)
    radii = np.array([a.radius for a in agents], dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    nbr_radius = max(0.65, 4.5 * float(np.max(radii)))
    close_cnt = (dist <= nbr_radius).sum(axis=1)
    contact_cnt = (dist <= (radii[:, None] + radii[None, :] + 0.05)).sum(axis=1)
    score = (close_cnt + 0.75 * contact_cnt) / 5.5
    return {a.id: float(np.clip(score[i], 0.0, 1.0)) for i, a in enumerate(agents)}


def _compute_direction_from_dist(dist: np.ndarray) -> np.ndarray:
    ny, nx = dist.shape
    pad = np.pad(dist, 1, constant_values=np.inf)
    vec = np.zeros((ny, nx, 2), dtype=float)
    for dj, di in _OFFSETS:
        nb = pad[1 + dj : 1 + dj + ny, 1 + di : 1 + di + nx]
        ok = np.isfinite(dist) & np.isfinite(nb)
        with np.errstate(invalid="ignore"):
            gain = np.where(ok, np.maximum(dist - nb, 0.0), 0.0)
        norm = math.hypot(dj, di)
        vec[..., 0] += gain * di / norm / norm
        vec[..., 1] += gain * dj / norm / norm
    n = np.linalg.norm(vec, axis=2)
    has = n > 1e-12
    vec[has] /= n[has][:, None]
    return vec


def _extract_nav_grid(sim: Simulation):
    """Zwraca (dist, dir_field, cell_size) dla aktualnego modelu i środowiska."""
    env = sim.env
    model = sim.model

    fields = getattr(model, "_fields", None)
    if isinstance(fields, dict) and fields:
        valid = [f for f in fields.values() if f is not None and hasattr(f, "dist")]
        if valid:
            cell = float(valid[0].h)
            dists = np.stack([f.dist for f in valid], axis=0)
            dirs = np.stack([f.dir for f in valid], axis=0)
            best_idx = np.argmin(dists, axis=0)
            jj, ii = np.indices(best_idx.shape)
            return dists[best_idx, jj, ii], dirs[best_idx, jj, ii], cell

    nav_field = getattr(model, "nav_field", None)
    if not (isinstance(nav_field, np.ndarray) and nav_field.size > 0):
        nav_field = getattr(model, "static_field", None)
    cell_size = getattr(model, "cell_size", None)
    if isinstance(nav_field, np.ndarray) and nav_field.size > 0 and cell_size:
        return nav_field, _compute_direction_from_dist(nav_field), float(cell_size)

    if not env.exits:
        return None

    cell = 0.25
    nx = max(1, int(math.ceil(env.width / cell)))
    ny = max(1, int(math.ceil(env.height / cell)))
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    best_d = np.full(len(pts), np.inf)
    best_vec = np.zeros((len(pts), 2), dtype=float)
    for ex in env.exits:
        ab = ex.b - ex.a
        t = np.clip(((pts - ex.a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        q = ex.a + t[:, None] * ab
        vec = q - pts
        d = np.linalg.norm(vec, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_vec[better] = vec[better]

    nrm = np.linalg.norm(best_vec, axis=1, keepdims=True)
    best_vec = np.where(nrm > 1e-9, best_vec / np.maximum(nrm, 1e-12), 0.0)
    return best_d.reshape(ny, nx), best_vec.reshape(ny, nx, 2), cell


def _build_nav_overlay(
    sim: Simulation,
    map_w: int,
    win_h: int,
    scale: int,
    x_offset: int,
    y_offset: int,
) -> pygame.Surface | None:
    data = _extract_nav_grid(sim)
    if data is None:
        return None

    dist, dir_field, cell = data
    ny, nx = dist.shape
    env = sim.env

    fin = np.isfinite(dist)
    max_d = float(np.max(dist[fin])) if fin.any() else 1.0
    max_d = max(max_d, 1e-6)
    norm = np.where(fin, np.clip(dist / max_d, 0.0, 1.0), 0.0)

    rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
    rgba[fin, 0] = (55 + 145 * norm[fin]).astype(np.uint8)
    rgba[fin, 1] = (185 - 65 * norm[fin]).astype(np.uint8)
    rgba[fin, 2] = (210 - 50 * norm[fin]).astype(np.uint8)
    rgba[fin, 3] = 70

    rgba = np.ascontiguousarray(rgba[::-1])
    grid_surf = pygame.image.frombuffer(rgba.tobytes(), (nx, ny), "RGBA")
    px_w = max(1, int(round(nx * cell * scale)))
    px_h = max(1, int(round(ny * cell * scale)))
    scaled = pygame.transform.scale(grid_surf, (px_w, px_h))

    overlay = pygame.Surface((map_w, win_h), pygame.SRCALPHA)
    clip_rect = pygame.Rect(
        x_offset,
        y_offset,
        max(1, int(round(env.width * scale))),
        max(1, int(round(env.height * scale))),
    )
    overlay.set_clip(clip_rect)
    top_y = int(round((env.height - ny * cell) * scale)) + y_offset
    overlay.blit(scaled, (x_offset, top_y))

    stride = max(1, int(round(0.75 / max(cell, 1e-6))))
    arrow_m = min(0.45, max(0.25, 0.42 * stride * cell))
    P = lambda p: (int(p[0] * scale) + x_offset, int((env.height - p[1]) * scale) + y_offset)

    start_idx = stride // 2
    for j in range(start_idx, ny, stride):
        for i in range(start_idx, nx, stride):
            if not fin[j, i]:
                continue
            dvec = dir_field[j, i]
            dlen = float(np.linalg.norm(dvec))
            if dlen < 1e-6:
                continue
            ux, uy = dvec[0] / dlen, dvec[1] / dlen
            cx, cy = (i + 0.5) * cell, (j + 0.5) * cell
            if cx > env.width or cy > env.height:
                continue
            p0 = P((cx - 0.35 * arrow_m * ux, cy - 0.35 * arrow_m * uy))
            p1 = P((cx + 0.65 * arrow_m * ux, cy + 0.65 * arrow_m * uy))
            pygame.draw.line(overlay, (55, 85, 130, 155), p0, p1, 1)

            px_perp, py_perp = -uy, ux
            h_len = 0.28 * arrow_m
            w_len = 0.18 * arrow_m
            w1 = P((
                cx + (0.65 * arrow_m - h_len) * ux + w_len * px_perp,
                cy + (0.65 * arrow_m - h_len) * uy + w_len * py_perp,
            ))
            w2 = P((
                cx + (0.65 * arrow_m - h_len) * ux - w_len * px_perp,
                cy + (0.65 * arrow_m - h_len) * uy - w_len * py_perp,
            ))
            pygame.draw.line(overlay, (55, 85, 130, 155), p1, w1, 1)
            pygame.draw.line(overlay, (55, 85, 130, 155), p1, w2, 1)

    overlay.set_clip(None)
    return overlay


def _draw_evac_chart(
    screen: pygame.Surface,
    rect: pygame.Rect,
    history: list[tuple[float, int]],
    current_t: float,
    total_agents: int,
    font_small: pygame.font.Font,
):
    pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=6)
    pygame.draw.rect(screen, (205, 212, 222), rect, 1, border_radius=6)

    title = font_small.render("Postęp ewakuacji", True, (50, 60, 75))
    screen.blit(title, (rect.x + 8, rect.y + 5))

    plot = pygame.Rect(rect.x + 30, rect.y + 24, rect.w - 40, rect.h - 42)
    if plot.w <= 10 or plot.h <= 10:
        return

    for frac in (0.25, 0.5, 0.75, 1.0):
        gy = int(plot.bottom - frac * plot.h)
        pygame.draw.line(screen, (236, 240, 246), (plot.x, gy), (plot.right, gy), 1)

    pygame.draw.line(screen, (175, 184, 198), (plot.x, plot.y), (plot.x, plot.bottom), 1)
    pygame.draw.line(screen, (175, 184, 198), (plot.x, plot.bottom), (plot.right, plot.bottom), 1)

    max_t = max(10.0, current_t)
    max_n = max(1, total_agents)

    pts = []
    for t_val, n_val in history:
        px = plot.x + int((t_val / max_t) * plot.w)
        py = plot.bottom - int((n_val / max_n) * plot.h)
        pts.append((px, py))

    if len(pts) >= 2:
        poly = [(pts[0][0], plot.bottom)] + pts + [(pts[-1][0], plot.bottom)]
        fill_surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        rel_poly = [(px - rect.x, py - rect.y) for px, py in poly]
        pygame.draw.polygon(fill_surf, (35, 175, 95, 48), rel_poly)
        screen.blit(fill_surf, rect.topleft)
        pygame.draw.aalines(screen, (25, 165, 85), False, pts)
    elif len(pts) == 1:
        pygame.draw.circle(screen, (25, 165, 85), pts[0], 2)

    lbl_top = font_small.render(str(total_agents), True, (115, 125, 140))
    lbl_zero = font_small.render("0", True, (115, 125, 140))
    lbl_t0 = font_small.render("0s", True, (115, 125, 140))
    lbl_tmax = font_small.render(f"{max_t:.0f}s", True, (115, 125, 140))
    screen.blit(lbl_top, (rect.x + 4, plot.y - 6))
    screen.blit(lbl_zero, (rect.x + 12, plot.bottom - 8))
    screen.blit(lbl_t0, (plot.x, plot.bottom + 2))
    screen.blit(lbl_tmax, (plot.right - lbl_tmax.get_width(), plot.bottom + 2))


def _fit_text(font: pygame.font.Font, text: str, max_w: int) -> str:
    if font.size(text)[0] <= max_w:
        return text
    s = text
    while len(s) > 4 and font.size("..." + s)[0] > max_w:
        s = s[1:]
    return "..." + s


def _draw_summary_modal(
    screen: pygame.Surface,
    map_rect: pygame.Rect,
    sim: Simulation,
    initial_total: int,
    short_scen_name: str,
    saved_csv_path: str | None,
    font_title: pygame.font.Font,
    font: pygame.font.Font,
    font_small: pygame.font.Font,
) -> dict[str, pygame.Rect]:
    dim = pygame.Surface((map_rect.w, map_rect.h), pygame.SRCALPHA)
    dim.fill((20, 26, 36, 95))
    screen.blit(dim, map_rect.topleft)

    mw = min(430, max(300, map_rect.w - 32))
    mh = 250 if saved_csv_path else 224
    mx = map_rect.x + (map_rect.w - mw) // 2
    my = map_rect.y + (map_rect.h - mh) // 2
    card = pygame.Rect(mx, my, mw, mh)

    pygame.draw.rect(screen, (255, 255, 255), card, border_radius=10)
    pygame.draw.rect(screen, (170, 180, 196), card, 2, border_radius=10)

    screen.blit(font_title.render("Ewakuacja zakończona!", True, (20, 145, 75)), (mx + 18, my + 14))
    screen.blit(font_small.render(f"{short_scen_name}  •  model: {sim.model.name}", True, (95, 105, 120)), (mx + 18, my + 40))

    summary = sim.summary()
    total_t = summary["evac_time_total"] or sim.t
    mean_t = summary["evac_time_mean"] or 0.0
    evac_n = summary["evacuated"]
    flow = (evac_n / total_t) if total_t > 1e-6 else 0.0

    rows = [
        f"Ewakuowani:         {evac_n} / {initial_total}",
        f"Czas całkowity:     {total_t:.2f} s",
        f"Średni czas:        {mean_t:.2f} s",
        f"Przepustowość śr.:  {flow:.2f} os./s",
    ]
    for i, line in enumerate(rows):
        screen.blit(font.render(line, True, (35, 40, 50)), (mx + 18, my + 68 + i * 24))

    if saved_csv_path:
        csv_text = _fit_text(font_small, f"Zapisano CSV: {saved_csv_path}", mw - 36)
        csv_lbl = font_small.render(csv_text, True, (45, 110, 195))
        screen.blit(csv_lbl, (mx + 18, my + 170))

    btn_y = card.bottom - 44
    bw_btn = (mw - 48) // 3
    rects = {
        "hide": pygame.Rect(mx + 16, btn_y, bw_btn, 30),
        "restart": pygame.Rect(mx + 24 + bw_btn, btn_y, bw_btn, 30),
        "menu": pygame.Rect(mx + 32 + 2 * bw_btn, btn_y, bw_btn, 30),
    }
    specs = [
        ("hide", "Podgląd", (232, 236, 242), (35, 40, 50)),
        ("restart", "Restart [R]", (45, 115, 220), (255, 255, 255)),
        ("menu", "Menu [ESC]", (220, 226, 235), (35, 40, 50)),
    ]
    for key, text, bg, fg in specs:
        r = rects[key]
        pygame.draw.rect(screen, bg, r, border_radius=6)
        pygame.draw.rect(screen, (180, 188, 200), r, 1, border_radius=6)
        lbl = font_small.render(text, True, fg)
        screen.blit(lbl, lbl.get_rect(center=r.center))

    return rects


def run_interactive_viewer(
    sim: Simulation,
    scenario_path: str,
    seed: int,
    scale: int = 55,
    fps: int = 60,
    model_params: dict | None = None,
) -> bool:
    pygame.init()
    env = sim.env

    sidebar_w = 310
    info = pygame.display.Info()
    max_scale = _max_scale_for(env.width, env.height, info.current_w, info.current_h, sidebar_w)
    scale = min(max(10, int(scale)), max_scale)

    raw_map_w = int(env.width * scale)
    raw_map_h = int(env.height * scale)
    map_w = max(raw_map_w, 420)
    win_h = max(raw_map_h, 600)
    win_w = map_w + sidebar_w

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Crowd Simulator — {Path(scenario_path).as_posix()}")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("segoeui", 18, bold=True)
    font = pygame.font.SysFont("segoeui", 15)
    font_small = pygame.font.SysFont("segoeui", 13)

    x_offset = (map_w - raw_map_w) // 2
    y_offset = (win_h - raw_map_h) // 2
    P = lambda p: (int(p[0] * scale) + x_offset, int((env.height - p[1]) * scale) + y_offset)

    paused = False
    running = True
    return_to_menu = False
    steps_per_frame = 1
    initial_total = len(sim.agents)

    color_mode_idx = 0  # 0: speed, 1: density, 2: solid
    show_nav = False
    show_trails = False
    show_vectors = True

    smooth_speeds: dict[int, float] = {}
    density_ratios: dict[int, float] = {}
    evac_history: list[tuple[float, int]] = [(0.0, 0)]
    last_chart_t = 0.0

    trail_surface = pygame.Surface((map_w, win_h), pygame.SRCALPHA)
    nav_surface: pygame.Surface | None = None
    last_nav_version = -1

    summary_dismissed = False
    saved_csv_path: str | None = None
    modal_buttons: dict[str, pygame.Rect] = {}
    buttons: dict[str, pygame.Rect] = {}

    def reset_simulation():
        nonlocal env, initial_total, nav_surface, last_nav_version
        nonlocal summary_dismissed, saved_csv_path, last_chart_t
        env_new, agents_new = load_scenario(scenario_path, seed=seed)
        sim.env = env_new
        sim.agents = agents_new
        sim.evacuated = []
        sim.t = 0.0
        sim.model.reset(sim.agents, sim.env)
        if sim.logger:
            sim.logger.rows.clear()
        env = sim.env
        initial_total = len(sim.agents)
        smooth_speeds.clear()
        density_ratios.clear()
        evac_history[:] = [(0.0, 0)]
        last_chart_t = 0.0
        trail_surface.fill((0, 0, 0, 0))
        nav_surface = None
        last_nav_version = -1
        summary_dismissed = False
        saved_csv_path = None

    def draw_button(
        rect: pygame.Rect,
        text: str,
        active: bool = False,
        color=(230, 234, 240),
        text_color=(30, 35, 45),
        small: bool = False,
    ):
        bg = (45, 115, 220) if active else color
        fg = (255, 255, 255) if active else text_color
        pygame.draw.rect(screen, bg, rect, border_radius=6)
        pygame.draw.rect(screen, (180, 188, 200), rect, 1, border_radius=6)
        lbl = (font_small if small else font).render(text, True, fg)
        screen.blit(lbl, lbl.get_rect(center=rect.center))

    p_obj = Path(scenario_path)
    short_scen_name = f"{p_obj.parent.name}/{p_obj.name}" if p_obj.parent.name != "scenarios" else p_obj.name

    while running:
        bx = map_w + 16
        bw = sidebar_w - 32

        buttons["pause"] = pygame.Rect(bx, 198, (bw - 8) // 2, 32)
        buttons["restart"] = pygame.Rect(bx + (bw - 8) // 2 + 8, 198, (bw - 8) // 2, 32)
        buttons["speed_down"] = pygame.Rect(bx, 238, 42, 28)
        buttons["speed_up"] = pygame.Rect(bx + bw - 42, 238, 42, 28)

        buttons["color_mode"] = pygame.Rect(bx, 298, bw - 58, 28)
        buttons["vectors"] = pygame.Rect(bx + bw - 50, 298, 50, 28)
        buttons["nav"] = pygame.Rect(bx, 358, (bw - 8) // 2, 28)
        buttons["trails"] = pygame.Rect(bx + (bw - 8) // 2 + 8, 358, (bw - 8) // 2, 28)
        buttons["menu"] = pygame.Rect(bx, win_h - 80, bw, 32)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                return_to_menu = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                    return_to_menu = True
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    steps_per_frame = min(20, steps_per_frame + 1)
                elif e.key == pygame.K_MINUS:
                    steps_per_frame = max(1, steps_per_frame - 1)
                elif e.key == pygame.K_r:
                    reset_simulation()
                elif e.key == pygame.K_n:
                    show_nav = not show_nav
                elif e.key == pygame.K_t:
                    show_trails = not show_trails
                elif e.key == pygame.K_c:
                    color_mode_idx = (color_mode_idx + 1) % len(COLOR_MODES)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                if sim.finished and not summary_dismissed and modal_buttons:
                    if modal_buttons["hide"].collidepoint(mx, my):
                        summary_dismissed = True
                        continue
                    if modal_buttons["restart"].collidepoint(mx, my):
                        reset_simulation()
                        continue
                    if modal_buttons["menu"].collidepoint(mx, my):
                        running = False
                        return_to_menu = True
                        continue

                if buttons["pause"].collidepoint(mx, my):
                    paused = not paused
                elif buttons["restart"].collidepoint(mx, my):
                    reset_simulation()
                elif buttons["speed_down"].collidepoint(mx, my):
                    steps_per_frame = max(1, steps_per_frame - 1)
                elif buttons["speed_up"].collidepoint(mx, my):
                    steps_per_frame = min(20, steps_per_frame + 1)
                elif buttons["color_mode"].collidepoint(mx, my):
                    color_mode_idx = (color_mode_idx + 1) % len(COLOR_MODES)
                elif buttons["vectors"].collidepoint(mx, my):
                    show_vectors = not show_vectors
                elif buttons["nav"].collidepoint(mx, my):
                    show_nav = not show_nav
                elif buttons["trails"].collidepoint(mx, my):
                    show_trails = not show_trails
                elif buttons["menu"].collidepoint(mx, my):
                    running = False
                    return_to_menu = True

        if not paused and not sim.finished:
            prev_positions = {ag.id: ag.pos.copy() for ag in sim.agents}
            steps_done = 0
            for _ in range(steps_per_frame):
                if not sim.finished:
                    sim.step()
                    steps_done += 1

            elapsed_dt = max(sim.dt * steps_done, 1e-6)
            for ag in sim.agents:
                prev_p = prev_positions.get(ag.id)
                if prev_p is None:
                    continue
                disp = float(np.linalg.norm(ag.pos - prev_p))
                inst_speed = disp / elapsed_dt
                if ag.id in smooth_speeds:
                    smooth_speeds[ag.id] = 0.35 * inst_speed + 0.65 * smooth_speeds[ag.id]
                else:
                    smooth_speeds[ag.id] = inst_speed

                if disp > 1e-3:
                    ratio = 1.0 - min(1.0, smooth_speeds[ag.id] / max(ag.desired_speed, 1e-6))
                    tc = _ratio_to_color(ratio)
                    pygame.draw.line(trail_surface, (*tc, 115), P(prev_p), P(ag.pos), 2)

            evac_now = len(sim.evacuated)
            if evac_now != evac_history[-1][1] or (sim.t - last_chart_t >= 0.25) or sim.finished:
                evac_history.append((sim.t, evac_now))
                last_chart_t = sim.t

        if sim.finished and sim.logger and saved_csv_path is None:
            sim.logger.save()
            saved_csv_path = sim.logger.path

        color_mode = COLOR_MODES[color_mode_idx]
        if color_mode == "density":
            density_ratios = _compute_density_ratios(sim.agents)

        cur_nav_version = getattr(sim.model, "nav_version", 0)
        if show_nav and (nav_surface is None or cur_nav_version != last_nav_version):
            nav_surface = _build_nav_overlay(sim, map_w, win_h, scale, x_offset, y_offset)
            last_nav_version = cur_nav_version

        screen.fill((248, 249, 252))

        for z in env.spawn_zones:
            top_left = P((z.x, z.y + z.h))
            rect = pygame.Rect(top_left[0], top_left[1], int(z.w * scale), int(z.h * scale))
            pygame.draw.rect(screen, (228, 238, 252), rect)

        if show_nav and nav_surface is not None:
            screen.blit(nav_surface, (0, 0))

        if show_trails:
            screen.blit(trail_surface, (0, 0))

        for obs in env.obstacles:
            top_left = P((obs["x"], obs["y"] + obs["h"]))
            rect = pygame.Rect(top_left[0], top_left[1], int(obs["w"] * scale), int(obs["h"] * scale))
            pygame.draw.rect(screen, (205, 210, 220), rect)

        for a, b in env.walls:
            pygame.draw.line(screen, (35, 40, 50), P(a), P(b), 3)

        for ex in env.exits:
            pygame.draw.line(screen, (20, 175, 85), P(ex.a), P(ex.b), 6)

        for ag in sim.agents:
            center = P(ag.pos)
            rad = max(3, int(ag.radius * scale))
            if color_mode == "speed":
                v_cur = smooth_speeds.get(ag.id, ag.desired_speed)
                ratio = 1.0 - min(1.0, v_cur / max(ag.desired_speed, 1e-6))
                fill_col = _ratio_to_color(ratio)
                edge_col = tuple(int(c * 0.6) for c in fill_col)
            elif color_mode == "density":
                ratio = density_ratios.get(ag.id, 0.0)
                fill_col = _ratio_to_color(ratio)
                edge_col = tuple(int(c * 0.6) for c in fill_col)
            else:
                fill_col = (35, 105, 225)
                edge_col = (20, 65, 155)

            pygame.draw.circle(screen, fill_col, center, rad)
            pygame.draw.circle(screen, edge_col, center, rad, 1)
            if show_vectors and ag.speed > 0.05:
                end_v = P(ag.pos + ag.vel * 0.25)
                pygame.draw.line(screen, (45, 55, 75), center, end_v, 2)

        if sim.finished and not summary_dismissed:
            modal_buttons = _draw_summary_modal(
                screen,
                pygame.Rect(0, 0, map_w, win_h),
                sim,
                initial_total,
                short_scen_name,
                saved_csv_path,
                font_title,
                font,
                font_small,
            )
        else:
            modal_buttons = {}

        # --- Panel boczny ---
        pygame.draw.rect(screen, (238, 241, 246), (map_w, 0, sidebar_w, win_h))
        pygame.draw.line(screen, (205, 212, 222), (map_w, 0), (map_w, win_h), 2)

        screen.blit(font_title.render("Panel Symulacji", True, (25, 30, 40)), (bx, 14))
        screen.blit(font_small.render(_fit_text(font_small, f"Scenariusz: {short_scen_name}", bw), True, (90, 100, 115)), (bx, 40))
        screen.blit(font_small.render(_fit_text(font_small, f"Model: {sim.model.name}", bw), True, (90, 100, 115)), (bx, 58))

        evac_count = len(sim.evacuated)
        in_count = len(sim.agents)
        pct = (evac_count / initial_total * 100) if initial_total else 100.0
        mean_str = (
            f"{sum(a.exit_time for a in sim.evacuated) / len(sim.evacuated):.1f} s"
            if sim.evacuated else "---"
        )

        stats = [
            f"Czas symulacji:  {sim.t:6.1f} s",
            f"W środku:        {in_count} / {initial_total}",
            f"Ewakuowani:      {evac_count} ({pct:.0f}%)",
            f"Śr. czas ewak.:  {mean_str}",
        ]
        for idx, line in enumerate(stats):
            screen.blit(font.render(line, True, (35, 40, 50)), (bx, 84 + idx * 22))

        bar_rect = pygame.Rect(bx, 176, bw, 10)
        pygame.draw.rect(screen, (210, 216, 226), bar_rect, border_radius=5)
        if pct > 0:
            fill_rect = pygame.Rect(bx, 176, int(bw * (pct / 100.0)), 10)
            pygame.draw.rect(screen, (25, 170, 85), fill_rect, border_radius=5)

        draw_button(buttons["pause"], "Wznów" if paused else "Pauza", active=paused)
        draw_button(buttons["restart"], "Restart [R]")

        draw_button(buttons["speed_down"], "-")
        spd_lbl = font.render(f"Tempo: {steps_per_frame}x", True, (35, 40, 50))
        screen.blit(spd_lbl, spd_lbl.get_rect(center=(bx + bw // 2, 252)))
        draw_button(buttons["speed_up"], "+")

        screen.blit(font_small.render("Warstwy i wizualizacja:", True, (90, 100, 115)), (bx, 276))
        draw_button(buttons["color_mode"], COLOR_MODE_LABELS[color_mode], small=True)
        draw_button(buttons["vectors"], "Wekt.", active=show_vectors, small=True)

        if color_mode in COLOR_LEGEND_LABELS:
            leg_rect = pygame.Rect(bx, 331, bw, 8)
            for ix in range(bw):
                c = _ratio_to_color(ix / max(1, bw - 1))
                pygame.draw.line(screen, c, (bx + ix, leg_rect.y), (bx + ix, leg_rect.bottom - 1))
            pygame.draw.rect(screen, (180, 188, 200), leg_rect, 1)
            l_low, l_mid, l_high = COLOR_LEGEND_LABELS[color_mode]
            lbl_l = font_small.render(l_low, True, (95, 105, 120))
            lbl_m = font_small.render(l_mid, True, (95, 105, 120))
            lbl_h = font_small.render(l_high, True, (95, 105, 120))
            screen.blit(lbl_l, (bx, 340))
            screen.blit(lbl_m, (bx + (bw - lbl_m.get_width()) // 2, 340))
            screen.blit(lbl_h, (bx + bw - lbl_h.get_width(), 340))

        draw_button(buttons["nav"], "Pole nawig. [N]", active=show_nav, small=True)
        draw_button(buttons["trails"], "Ślady [T]", active=show_trails, small=True)

        chart_h = min(145, max(90, win_h - 398 - 92))
        chart_rect = pygame.Rect(bx, 398, bw, chart_h)
        _draw_evac_chart(screen, chart_rect, evac_history, sim.t, initial_total, font_small)

        draw_button(buttons["menu"], "← Zmień scenariusz [ESC]", color=(220, 224, 232))

        hints = ["[SPACJA] Pauza  [+/-] Tempo  [C] Kolor", "[N] Pole  [T] Ślady  [R] Restart  [ESC] Menu"]
        for idx, h in enumerate(hints):
            screen.blit(font_small.render(h, True, (110, 120, 135)), (bx, win_h - 42 + idx * 17))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    return return_to_menu


class SimulatorLauncher:

    def __init__(self, scenarios_dir: str = "scenarios", last_config: dict | None = None):
        self.scenarios_dir = Path(scenarios_dir)
        self.last_config = last_config or {}
        self.launch_config: dict | None = None

        self.root = tk.Tk()
        self.root.title("Crowd Simulator — Konfiguracja")
        self.root.geometry("500x495")
        self.root.resizable(False, False)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        self._build_ui()

    def _find_scenarios(self) -> list[str]:
        if not self.scenarios_dir.exists():
            return []
        files = sorted(self.scenarios_dir.rglob("*.json"))
        return [p.as_posix() for p in files]

    def _build_ui(self):
        pad = {"padx": 14, "pady": 5}
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Crowd Simulator", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=14, pady=(2, 8))

        ttk.Label(frm, text="Scenariusz:").pack(anchor="w", padx=14)
        scen_row = ttk.Frame(frm)
        scen_row.pack(fill="x", **pad)

        self.scenarios = self._find_scenarios()
        default_scen = self.last_config.get("scenario_path") or (self.scenarios[0] if self.scenarios else "")
        self.scenario_var = tk.StringVar(value=default_scen)
        self.scenario_combo = ttk.Combobox(scen_row, textvariable=self.scenario_var, values=self.scenarios, state="readonly")
        self.scenario_combo.pack(side="left", fill="x", expand=True)
        self.scenario_combo.bind("<<ComboboxSelected>>", self._on_selection_change)

        ttk.Button(scen_row, text="...", width=3, command=self._browse_scenario).pack(side="right", padx=(6, 0))

        ttk.Label(frm, text="Model ruchu tłumu:").pack(anchor="w", padx=14)
        default_model = self.last_config.get("model_name") or list(MODEL_REGISTRY.keys())[0]
        self.model_var = tk.StringVar(value=default_model)
        model_combo = ttk.Combobox(frm, textvariable=self.model_var, values=list(MODEL_REGISTRY.keys()), state="readonly")
        model_combo.pack(fill="x", **pad)
        model_combo.bind("<<ComboboxSelected>>", self._on_selection_change)

        grid = ttk.Frame(frm)
        grid.pack(fill="x", **pad)

        ttk.Label(grid, text="Seed:").grid(row=0, column=0, sticky="w")
        self.seed_var = tk.StringVar(value=str(self.last_config.get("seed", 0)))
        ttk.Entry(grid, textvariable=self.seed_var, width=10).grid(row=1, column=0, padx=(0, 12), sticky="w")

        ttk.Label(grid, text="Krok dt [s]:").grid(row=0, column=1, sticky="w")
        self.dt_var = tk.StringVar(value=str(self.last_config.get("dt", 0.05)))
        ttk.Entry(grid, textvariable=self.dt_var, width=10).grid(row=1, column=1, padx=(0, 12), sticky="w")

        ttk.Label(grid, text="Skala [px/m]:").grid(row=0, column=2, sticky="w")
        self.scale_var = tk.StringVar(value=str(self.last_config.get("scale", 55)))
        ttk.Entry(grid, textvariable=self.scale_var, width=8).grid(row=1, column=2, sticky="w")

        self.scale_hint_var = tk.StringVar(value="")
        ttk.Label(grid, textvariable=self.scale_hint_var, foreground="#5a6473").grid(
            row=1, column=3, padx=(8, 0), sticky="w"
        )

        # --- Edycja parametrów modelu ---
        params_hdr = ttk.Frame(frm)
        params_hdr.pack(fill="x", padx=14, pady=(6, 2))
        ttk.Label(params_hdr, text="Parametry modelu:").pack(side="left")

        self.params_status_var = tk.StringVar(value="")
        ttk.Label(params_hdr, textvariable=self.params_status_var, foreground="#2d6cdf").pack(side="left", padx=(6, 0))

        ttk.Button(params_hdr, text="Domyślne", command=self._fill_default_params).pack(side="right")
        ttk.Button(params_hdr, text="Ze scenariusza", command=self._load_scenario_params).pack(side="right", padx=(0, 4))

        self.params_text = tk.Text(frm, height=5, width=54, font=("Consolas", 9))
        self.params_text.pack(fill="x", padx=14, pady=(0, 6))

        self.save_csv_var = tk.BooleanVar(value=self.last_config.get("save_csv", False))
        ttk.Checkbutton(
            frm,
            text="Zapisz trajektorie do pliku CSV",
            variable=self.save_csv_var,
        ).pack(anchor="w", padx=14, pady=4)

        ttk.Button(frm, text="Uruchom symulację", command=self._on_start).pack(fill="x", padx=14, pady=(8, 4), ipady=6)

        self._refresh_scale_limit(adjust_value=False)
        if "model_params" in self.last_config:
            self._set_params_text(self.last_config["model_params"])
            scen_p = _get_model_params(self.scenario_var.get(), self.model_var.get())
            if self.last_config["model_params"] != scen_p:
                self.params_status_var.set("zmodyfikowane")
            elif scen_p:
                self.params_status_var.set("ze scenariusza")
            else:
                self.params_status_var.set("domyślne")
        else:
            self._load_scenario_params()

    def _current_max_scale(self) -> int:
        cfg = _read_scenario_json(self.scenario_var.get())
        w = float(cfg.get("width", 12.0))
        h = float(cfg.get("height", 10.0))
        return _max_scale_for(w, h, self.screen_w, self.screen_h)

    def _refresh_scale_limit(self, adjust_value: bool = True):
        max_sc = self._current_max_scale()
        self.scale_hint_var.set(f"maks. {max_sc}")
        try:
            cur = int(self.scale_var.get())
        except Exception:
            cur = 55
        if adjust_value:
            self.scale_var.set(str(min(55, max_sc) if cur > max_sc else cur))
        elif cur > max_sc:
            self.scale_var.set(str(max_sc))

    def _set_params_text(self, params: dict):
        self.params_text.delete("1.0", tk.END)
        self.params_text.insert("1.0", json.dumps(params, indent=2, ensure_ascii=False))

    def _load_scenario_params(self):
        params = _get_model_params(self.scenario_var.get(), self.model_var.get())
        self._set_params_text(params)
        self.params_status_var.set("ze scenariusza" if params else "domyślne")

    def _fill_default_params(self):
        defaults = _get_default_model_params(self.model_var.get())
        self._set_params_text(defaults)
        self.params_status_var.set("szablon domyślnych")

    def _on_selection_change(self, _event=None):
        self._refresh_scale_limit(adjust_value=True)
        self._load_scenario_params()

    def _browse_scenario(self):
        path = filedialog.askopenfilename(initialdir=self.scenarios_dir, filetypes=[("JSON", "*.json")])
        if path:
            try:
                rel = Path(path).relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel = Path(path).as_posix()
            self.scenario_var.set(rel)
            self._on_selection_change()

    def _on_start(self):
        scenario_path = self.scenario_var.get()
        if not scenario_path or not os.path.exists(scenario_path):
            messagebox.showerror("Błąd", "Wybierz poprawny plik scenariusza JSON.")
            return

        try:
            seed = int(self.seed_var.get())
            dt = float(self.dt_var.get())
            if dt <= 0:
                raise ValueError("Krok dt musi być dodatni.")
            scale = int(self.scale_var.get())
            max_sc = self._current_max_scale()
            if scale > max_sc:
                scale = max_sc
                self.scale_var.set(str(scale))
            if scale < 10:
                scale = 10
        except Exception as exc:
            messagebox.showerror("Błąd parametrów", f"Niepoprawne dane wejściowe:\n{exc}")
            return

        raw_params = self.params_text.get("1.0", tk.END).strip()
        try:
            model_params = json.loads(raw_params) if raw_params else {}
            if not isinstance(model_params, dict):
                raise ValueError("Parametry modelu muszą być obiektem JSON.")
        except Exception as exc:
            messagebox.showerror("Błąd parametrów modelu", f"Niepoprawny JSON parametrów:\n{exc}")
            return

        self.launch_config = {
            "scenario_path": scenario_path,
            "model_name": self.model_var.get(),
            "seed": seed,
            "dt": dt,
            "scale": scale,
            "save_csv": self.save_csv_var.get(),
            "model_params": model_params,
        }
        self.root.destroy()

    def get_config(self) -> dict | None:
        self.root.mainloop()
        return self.launch_config


def launch_gui():
    last_config = None
    while True:
        launcher = SimulatorLauncher(last_config=last_config)
        cfg = launcher.get_config()
        if cfg is None:
            break

        last_config = cfg
        params = dict(cfg.get("model_params") or {})
        csv_path = _make_csv_path(cfg["scenario_path"], cfg["model_name"]) if cfg["save_csv"] else None
        logger = TrajectoryLogger(csv_path) if csv_path else None

        env, agents = load_scenario(cfg["scenario_path"], seed=cfg["seed"])
        model = create_model(cfg["model_name"], **params)
        sim = Simulation(env, agents, model, dt=cfg["dt"], logger=logger, seed=cfg["seed"])

        go_back = run_interactive_viewer(
            sim,
            scenario_path=cfg["scenario_path"],
            seed=cfg["seed"],
            scale=cfg["scale"],
            model_params=params,
        )
        print(sim.summary())
        if logger:
            logger.save()

        if not go_back:
            break
