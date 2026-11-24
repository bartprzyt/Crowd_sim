"""Graficzny interfejs użytkownika (Launcher + rozszerzony podgląd Pygame)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pygame

from .scenario import load_scenario
from .simulation import Simulation
from .logger import TrajectoryLogger
from .models import create_model, MODEL_REGISTRY


def _get_model_params(scenario_path: str, model_name: str) -> dict:
    try:
        with open(scenario_path, encoding="utf-8") as f:
            return dict(json.load(f).get("model_params", {}).get(model_name, {}))
    except Exception:
        return {}


def run_interactive_viewer(sim: Simulation, scenario_path: str, seed: int, scale: int = 55, fps: int = 60) -> bool:
    pygame.init()
    env = sim.env

    sidebar_w = 290
    map_w = int(env.width * scale)
    map_h = max(int(env.height * scale), 420)
    win_w = map_w + sidebar_w
    win_h = map_h

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Crowd Simulator — {Path(scenario_path).as_posix()}")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("segoeui", 18, bold=True)
    font = pygame.font.SysFont("segoeui", 15)
    font_small = pygame.font.SysFont("segoeui", 13)

    y_offset = (win_h - int(env.height * scale)) // 2
    P = lambda p: (int(p[0] * scale), int((env.height - p[1]) * scale) + y_offset)

    paused = False
    running = True
    return_to_menu = False
    steps_per_frame = 1
    initial_total = len(sim.agents)

    buttons = {}

    def draw_button(rect: pygame.Rect, text: str, active: bool = False, color=(230, 234, 240), text_color=(30, 35, 45)):
        bg = (45, 115, 220) if active else color
        fg = (255, 255, 255) if active else text_color
        pygame.draw.rect(screen, bg, rect, border_radius=6)
        pygame.draw.rect(screen, (180, 188, 200), rect, 1, border_radius=6)
        lbl = font.render(text, True, fg)
        screen.blit(lbl, lbl.get_rect(center=rect.center))

    p_obj = Path(scenario_path)
    short_scen_name = f"{p_obj.parent.name}/{p_obj.name}" if p_obj.parent.name != "scenarios" else p_obj.name

    while running:
        bx = map_w + 16
        bw = sidebar_w - 32

        buttons["pause"] = pygame.Rect(bx, 215, (bw - 8) // 2, 34)
        buttons["restart"] = pygame.Rect(bx + (bw - 8) // 2 + 8, 215, (bw - 8) // 2, 34)
        buttons["speed_down"] = pygame.Rect(bx, 258, 42, 30)
        buttons["speed_up"] = pygame.Rect(bx + bw - 42, 258, 42, 30)
        buttons["menu"] = pygame.Rect(bx, win_h - 88, bw, 34)

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
                    env_new, agents_new = load_scenario(scenario_path, seed=seed)
                    sim.env = env_new
                    sim.agents = agents_new
                    sim.evacuated = []
                    sim.t = 0.0
                    sim.model.reset(sim.agents, sim.env)
                    initial_total = len(sim.agents)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                if buttons["pause"].collidepoint(mx, my):
                    paused = not paused
                elif buttons["restart"].collidepoint(mx, my):
                    env_new, agents_new = load_scenario(scenario_path, seed=seed)
                    sim.env = env_new
                    sim.agents = agents_new
                    sim.evacuated = []
                    sim.t = 0.0
                    sim.model.reset(sim.agents, sim.env)
                    initial_total = len(sim.agents)
                elif buttons["speed_down"].collidepoint(mx, my):
                    steps_per_frame = max(1, steps_per_frame - 1)
                elif buttons["speed_up"].collidepoint(mx, my):
                    steps_per_frame = min(20, steps_per_frame + 1)
                elif buttons["menu"].collidepoint(mx, my):
                    running = False
                    return_to_menu = True

        if not paused and not sim.finished:
            for _ in range(steps_per_frame):
                if not sim.finished:
                    sim.step()

        screen.fill((248, 249, 252))

        for z in env.spawn_zones:
            top_left = P((z.x, z.y + z.h))
            rect = pygame.Rect(top_left[0], top_left[1], int(z.w * scale), int(z.h * scale))
            pygame.draw.rect(screen, (228, 238, 252), rect)

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
            pygame.draw.circle(screen, (35, 105, 225), center, rad)
            pygame.draw.circle(screen, (20, 65, 155), center, rad, 1)
            if ag.speed > 0.05:
                end_v = P(ag.pos + ag.vel * 0.25)
                pygame.draw.line(screen, (220, 60, 60), center, end_v, 2)

        pygame.draw.rect(screen, (238, 241, 246), (map_w, 0, sidebar_w, win_h))
        pygame.draw.line(screen, (205, 212, 222), (map_w, 0), (map_w, win_h), 2)

        screen.blit(font_title.render("Panel Symulacji", True, (25, 30, 40)), (bx, 16))
        screen.blit(font_small.render(f"Scenariusz: {short_scen_name}", True, (90, 100, 115)), (bx, 42))
        screen.blit(font_small.render(f"Model:      {sim.model.name}", True, (90, 100, 115)), (bx, 60))

        evac_count = len(sim.evacuated)
        in_count = len(sim.agents)
        pct = (evac_count / initial_total * 100) if initial_total else 100.0

        stats = [
            f"Czas symulacji:  {sim.t:6.1f} s",
            f"W środku:        {in_count} / {initial_total}",
            f"Ewakuowani:      {evac_count} ({pct:.0f}%)",
        ]
        if sim.evacuated:
            mean_t = sum(a.exit_time for a in sim.evacuated) / len(sim.evacuated)
            stats.append(f"Śr. czas ewakuacji: {mean_t:.1f} s")

        for idx, line in enumerate(stats):
            screen.blit(font.render(line, True, (35, 40, 50)), (bx, 92 + idx * 24))

        bar_rect = pygame.Rect(bx, 192, bw, 10)
        pygame.draw.rect(screen, (210, 216, 226), bar_rect, border_radius=5)
        if pct > 0:
            fill_rect = pygame.Rect(bx, 192, int(bw * (pct / 100.0)), 10)
            pygame.draw.rect(screen, (25, 170, 85), fill_rect, border_radius=5)

        draw_button(buttons["pause"], "Wznów" if paused else "Pauza", active=paused)
        draw_button(buttons["restart"], "Restart [R]")

        draw_button(buttons["speed_down"], "-")
        spd_lbl = font.render(f"Tempo: {steps_per_frame}x", True, (35, 40, 50))
        screen.blit(spd_lbl, spd_lbl.get_rect(center=(bx + bw // 2, 273)))
        draw_button(buttons["speed_up"], "+")

        draw_button(buttons["menu"], "← Zmień scenariusz [ESC]", color=(220, 224, 232))

        hints = ["[SPACJA] Pauza   [+/-] Tempo", "[R] Restart       [ESC] Menu"]
        for idx, h in enumerate(hints):
            screen.blit(font_small.render(h, True, (110, 120, 135)), (bx, win_h - 46 + idx * 18))

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
        self.root.geometry("450x320")
        self.root.resizable(False, False)

        self._build_ui()

    def _find_scenarios(self) -> list[str]:
        if not self.scenarios_dir.exists():
            return []
        files = sorted(self.scenarios_dir.rglob("*.json"))
        return [p.as_posix() for p in files]

    def _build_ui(self):
        pad = {"padx": 14, "pady": 6}
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Crowd Simulator", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=14, pady=(4, 10))

        ttk.Label(frm, text="Scenariusz (w tym podfoldery scenarios/):").pack(anchor="w", padx=14)
        scen_row = ttk.Frame(frm)
        scen_row.pack(fill="x", **pad)

        self.scenarios = self._find_scenarios()
        default_scen = self.last_config.get("scenario_path") or (self.scenarios[0] if self.scenarios else "")
        self.scenario_var = tk.StringVar(value=default_scen)
        self.scenario_combo = ttk.Combobox(scen_row, textvariable=self.scenario_var, values=self.scenarios, state="readonly")
        self.scenario_combo.pack(side="left", fill="x", expand=True)

        ttk.Button(scen_row, text="...", width=3, command=self._browse_scenario).pack(side="right", padx=(6, 0))

        ttk.Label(frm, text="Model ruchu tłumu:").pack(anchor="w", padx=14)
        default_model = self.last_config.get("model_name") or list(MODEL_REGISTRY.keys())[0]
        self.model_var = tk.StringVar(value=default_model)
        model_combo = ttk.Combobox(frm, textvariable=self.model_var, values=list(MODEL_REGISTRY.keys()), state="readonly")
        model_combo.pack(fill="x", **pad)

        grid = ttk.Frame(frm)
        grid.pack(fill="x", **pad)

        ttk.Label(grid, text="Seed:").grid(row=0, column=0, sticky="w")
        self.seed_var = tk.StringVar(value=str(self.last_config.get("seed", 0)))
        ttk.Entry(grid, textvariable=self.seed_var, width=10).grid(row=1, column=0, padx=(0, 12))

        ttk.Label(grid, text="Krok dt [s]:").grid(row=0, column=1, sticky="w")
        self.dt_var = tk.StringVar(value=str(self.last_config.get("dt", 0.05)))
        ttk.Entry(grid, textvariable=self.dt_var, width=10).grid(row=1, column=1, padx=(0, 12))

        ttk.Label(grid, text="Skala [px/m]:").grid(row=0, column=2, sticky="w")
        self.scale_var = tk.StringVar(value=str(self.last_config.get("scale", 55)))
        ttk.Entry(grid, textvariable=self.scale_var, width=10).grid(row=1, column=2)

        self.save_csv_var = tk.BooleanVar(value=self.last_config.get("save_csv", False))
        ttk.Checkbutton(frm, text="Zapisz trajektorie do output.csv po zakończeniu", variable=self.save_csv_var).pack(anchor="w", padx=14, pady=6)

        ttk.Button(frm, text="Uruchom symulację", command=self._on_start).pack(fill="x", padx=14, pady=(10, 4), ipady=6)

    def _browse_scenario(self):
        path = filedialog.askopenfilename(initialdir=self.scenarios_dir, filetypes=[("JSON", "*.json")])
        if path:
            try:
                rel = Path(path).relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel = Path(path).as_posix()
            self.scenario_var.set(rel)

    def _on_start(self):
        scenario_path = self.scenario_var.get()
        if not scenario_path or not os.path.exists(scenario_path):
            messagebox.showerror("Błąd", "Wybierz poprawny plik scenariusza JSON.")
            return

        try:
            seed = int(self.seed_var.get())
            dt = float(self.dt_var.get())
            scale = int(self.scale_var.get())
        except Exception as exc:
            messagebox.showerror("Błąd parametrów", f"Niepoprawne dane wejściowe:\n{exc}")
            return

        self.launch_config = {
            "scenario_path": scenario_path,
            "model_name": self.model_var.get(),
            "seed": seed,
            "dt": dt,
            "scale": scale,
            "save_csv": self.save_csv_var.get(),
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
        params = _get_model_params(cfg["scenario_path"], cfg["model_name"])
        logger = TrajectoryLogger("output.csv") if cfg["save_csv"] else None

        env, agents = load_scenario(cfg["scenario_path"], seed=cfg["seed"])
        model = create_model(cfg["model_name"], **params)
        sim = Simulation(env, agents, model, dt=cfg["dt"], logger=logger, seed=cfg["seed"])

        go_back = run_interactive_viewer(
            sim,
            scenario_path=cfg["scenario_path"],
            seed=cfg["seed"],
            scale=cfg["scale"],
        )
        print(sim.summary())
        if logger:
            logger.save()

        if not go_back:
            break