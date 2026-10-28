
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
file_path = DATA_DIR / "data.txt"

FPS = 25.0
DT = 1.0 / FPS

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=["PersID", "Frame", "X", "Y", "Z"],
)

print(df.head())
print("Wczytano rekordów:", len(df))

df = df.dropna(subset=["PersID", "Frame", "X", "Y", "Z"])

df = df.sort_values(["PersID", "Frame"]).reset_index(drop=True)

df["dx"] = df.groupby("PersID")["X"].diff()
df["dy"] = df.groupby("PersID")["Y"].diff()
df["dt"] = df.groupby("PersID")["Frame"].diff() / FPS

df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2) / df["dt"]
df.loc[df["dt"] <= 0, "speed"] = np.nan

print("Liczba rekordów:", len(df))
print("Liczba pieszych:", df["PersID"].nunique())
print("Zakres klatek:", df["Frame"].min(), "-", df["Frame"].max())
print("Średnia prędkość:", round(df["speed"].mean(), 3))
print("Mediana prędkości:", round(df["speed"].median(), 3))

df.to_csv(DATA_DIR / "real_data_with_speed.csv", index=False)
