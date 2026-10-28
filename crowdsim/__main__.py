"""Uruchomienie: python -m crowdsim --scenario scenarios/room_evacuation.json --model simple"""
import argparse
from .scenario import load_scenario
from .simulation import Simulation
from .logger import TrajectoryLogger
from .models import create_model, MODEL_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/room_evacuation.json")
    ap.add_argument("--model", default="simple", choices=list(MODEL_REGISTRY))
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=600.0)
    ap.add_argument("--headless", action="store_true", help="bez okna Pygame")
    ap.add_argument("--out", default=None, help="plik CSV z trajektoriami")
    a = ap.parse_args()

    env, agents = load_scenario(a.scenario, seed=a.seed)
    logger = TrajectoryLogger(a.out) if a.out else None
    sim = Simulation(env, agents, create_model(a.model), dt=a.dt, logger=logger, seed=a.seed)
    if a.headless:
        print(sim.run(a.t_max))
    else:
        from .viewer import run_viewer
        run_viewer(sim)
        print(sim.summary())
    if logger:
        logger.save()


if __name__ == "__main__":
    main()
