
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "real_data_with_speed.csv"

df = pd.read_csv(CSV_PATH)

valid = df.dropna(subset=["speed"])

print("=== STATYSTYKI RZECZYWISTEGO RUCHU ===")
print("Liczba pomiarów:", len(df))
print("Liczba pieszych:", df["PersID"].nunique())
print("Średnia prędkość:", round(valid["speed"].mean(), 3))
print("Mediana prędkości:", round(valid["speed"].median(), 3))
print("Odchylenie standardowe prędkości:",
      round(valid["speed"].std(), 3))

# Wykres trajektorii wszystkich pieszych
plt.figure(figsize=(10, 6))

for person_id, person in df.groupby("PersID"):
    plt.plot(person["X"], person["Y"], linewidth=1)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Rzeczywiste trajektorie pieszych")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()

plt.savefig(DATA_DIR / "trajectories.png", dpi=150)
plt.close()

# Histogram prędkości
plt.figure(figsize=(8, 5))

plt.hist(valid["speed"], bins=30)

plt.xlabel("Prędkość")
plt.ylabel("Liczba pomiarów")
plt.title("Rozkład prędkości pieszych")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(DATA_DIR / "speed_histogram.png", dpi=150)
plt.close()

# Średnia prędkość każdej osoby
person_speeds = valid.groupby("PersID")["speed"].mean()

person_speeds.to_csv(DATA_DIR / "person_speeds.csv",
                     header=["mean_speed"])

print("\nZapisano wykresy:")
print(DATA_DIR / "trajectories.png")
print(DATA_DIR / "speed_histogram.png")
print("Zapisano średnie prędkości osób:",
      DATA_DIR / "person_speeds.csv")
