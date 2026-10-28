"""Wizualizacja Pygame (spacja = pauza, +/- = prędkość, ESC = wyjście)."""
import pygame


def run_viewer(sim, scale=60, fps=60, steps_per_frame=1):
    pygame.init()
    env = sim.env
    screen = pygame.display.set_mode((int(env.width * scale), int(env.height * scale)))
    pygame.display.set_caption("Crowd Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    P = lambda p: (int(p[0] * scale), int((env.height - p[1]) * scale))
    paused, running = False, True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    steps_per_frame += 1
                elif e.key == pygame.K_MINUS:
                    steps_per_frame = max(1, steps_per_frame - 1)
        if not paused and not sim.finished:
            for _ in range(steps_per_frame):
                sim.step()
        screen.fill((250, 250, 250))
        for a, b in env.walls:
            pygame.draw.line(screen, (30, 30, 30), P(a), P(b), 3)
        for ex in env.exits:
            pygame.draw.line(screen, (0, 170, 0), P(ex.a), P(ex.b), 5)
        for ag in sim.agents:
            pygame.draw.circle(screen, (30, 100, 220), P(ag.pos), max(2, int(ag.radius * scale)))
        txt = f"model={sim.model.name}  t={sim.t:.1f}s  w srodku={len(sim.agents)}  ewak.={len(sim.evacuated)}"
        screen.blit(font.render(txt, True, (0, 0, 0)), (8, 8))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
