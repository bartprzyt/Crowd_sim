"""Wspólny format wyjściowy: t, id, x, y, vx, vy (CSV)."""
import csv


class TrajectoryLogger:
    HEADER = ["t", "id", "x", "y", "vx", "vy"]

    def __init__(self, path=None):
        self.rows = []
        self.path = path

    def log(self, t, agents):
        for a in agents:
            self.rows.append((round(t, 4), a.id, a.pos[0], a.pos[1], a.vel[0], a.vel[1]))

    def save(self, path=None):
        path = path or self.path
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.HEADER)
            w.writerows(self.rows)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.HEADER)
