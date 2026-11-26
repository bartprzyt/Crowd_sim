import argparse
import json
import sys
from .scenario import load_scenario
from .simulation import Simulation
from .logger import TrajectoryLogger
from .models import create_model, MODEL_REGISTRY


def main():
    if len(sys.argv) == 1:
        from .gui import launch_gui
        launch_gui()
        return
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="uruchom graficzny launcher")
    ap.add_argument("--scenario", default="scenarios/room_evacuation.json")
    ap.add_argument("--model", default="simple", choices=list(MODEL_REGISTRY))
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=600.0)
    ap.add_argument("--headless", action="store_true", help="bez okna Pygame")
    ap.add_argument("--out", default=None, help="plik CSV z trajektoriami")
    ap.add_argument("--params", default=None,
                    help='parametry modelu jako JSON, np. \'{"repulsion_range": 0.08}\'')
    ap.add_argument("--params-file", default=None, help="plik JSON z parametrami modelu")
    a = ap.parse_args()

    if a.gui:
        from .gui import launch_gui
        launch_gui()
        return

    with open(a.scenario, encoding="utf-8") as f:
        params = dict(json.load(f).get("model_params", {}).get(a.model, {}))
    if a.params_file:
        with open(a.params_file, encoding="utf-8") as f:
            params.update(json.load(f))
    if a.params:
        params.update(json.loads(a.params))

    env, agents = load_scenario(a.scenario, seed=a.seed)
    print(f"Agentów: {len(agents)}, model: {a.model}, parametry: {params or 'domyślne'}")
    logger = TrajectoryLogger(a.out) if a.out else None
    model = create_model(a.model, **params)
    sim = Simulation(env, agents, model, dt=a.dt, logger=logger, seed=a.seed)
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