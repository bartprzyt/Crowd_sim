from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
file_path = DATA_DIR / "data.txt"

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=["PersID", "Frame", "X", "Y", "Z"]
)

min_x, max_x = df["X"].min(), df["X"].max()
min_y, max_y = df["Y"].min(), df["Y"].max()

margin = 0.5
width = float((max_x - min_x) + 2 * margin)
height = float((max_y - min_y) + 2 * margin)

exit_x = width - margin
exit_width = 1.2
exit_y_center = height / 2.0
exit_y_start = max(margin, exit_y_center - exit_width / 2.0)
exit_y_end = min(height - margin, exit_y_center + exit_width / 2.0)

first_frame = df.groupby("PersID").first().reset_index()

walls = [
    [[0.0, 0.0], [width, 0.0]],
    [[0.0, height], [width, height]],
    [[0.0, 0.0], [0.0, height]],
    [[width, 0.0], [width, round(exit_y_start, 4)]],
    [[width, round(exit_y_end, 4)], [width, height]]
]

offset_x = -min_x + margin
offset_y = -min_y + margin

scenario = {
    "width": round(width, 4),
    "height": round(height, 4),
    "walls": walls,
    "obstacles": [],
    "exits": [
        [
            [round(exit_x, 4), round(margin, 4)],
            [round(exit_x, 4), round(height - margin, 4)]
        ]
    ],
    "spawn_zones": [
        {
            "x": round(margin, 4),
            "y": round(margin, 4),
            "w": round(width - 2 * margin, 4),
            "h": round(height - 2 * margin, 4),
            "n": len(first_frame),
            "goal_exit": 0
        }
    ]
}

output_dir = DATA_DIR / "scenarios"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "real_data_scenario.json", "w", encoding="utf-8") as f:
    json.dump(scenario, f, indent=4)
