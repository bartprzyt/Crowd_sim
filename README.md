# Crowd Simulator

Symulator dynamiki tłumu i ewakuacji z różnymi modelami dyskretnymi i ciągłymi.

## Uruchomienie programu

```bash
python -m crowdsim --scenario <ścieżka do scenariusza> --model <nazwa modelu>
```

Dostępne modele:
- `simple` – ruch prostoliniowy w stronę celu z detekcją kolizji ze ścianami
- `cellular` (alias `ca`) – automat komórkowy z polem statycznym (Dijkstra) i dynamicznym
- `social_force` – model sił społecznych Helbinga
- `social_force2` – rozszerzony model sił społecznych z polem nawigacyjnym wokół przeszkód

Opcje CLI:
- `--headless` – uruchomienie symulacji bez okna graficznego (tryb wsadowy/benchmark)
- `--dt <krok>` – krok czasowy symulacji (domyślnie `0.05`)
- `--seed <int>` – ziarno generatora losowego
- `--t-max <sekundy>` – maksymalny czas symulacji (domyślnie `600.0`)
- `--out <plik.csv>` – zapis trajektorii agentów do pliku CSV

Przykłady:
```bash
python -m crowdsim --scenario scenarios/narrow_door.json --model cellular
python -m crowdsim --scenario scenarios/room_with_obstacles.json --model social_force2
python -m crowdsim --scenario scenarios/classroom_evacuation.json --model cellular
python -m crowdsim --scenario scenarios/corner_corridor.json --model cellular
python -m crowdsim --scenario scenarios/column_before_exit.json --model social_force2
python -m crowdsim --scenario crowdsim/data/scenarios/real_data_scenario.json --model simple
```

---

## Dostępne scenariusze (`scenarios/`)

### Scenariusze ewakuacji i wąskich gardeł:
1. **`scenarios/narrow_door.json`** – sala 10x10 m z pojedynczymi wąskimi drzwiami (0.8 m); 50 agentów. Klasyczny test zatoru przy drzwiach.
2. **`scenarios/room_evacuation.json`** – standardowy pokój 12x8 m z wyjściem o szerokości 2.0 m; 20 agentów.
3. **`scenarios/funnel_bottleneck.json`** – lejek ze skośnymi ścianami płynnie zwężającymi korytarz do wąskiego gardła (1.6 m); 40 agentów. Bada kompresję i płynność przepływu w porównaniu do nagłych przewężeń.
4. **`scenarios/choke_corridor.json`** – korytarz z przewężeniem pośrodku utworzonym przez dwie przeszkody wystające ze ścian; 35 agentów.
5. **`scenarios/column_before_exit.json`** – sala z kolumną/filarem antypanicznym umieszczonym tuż przed wąskim wyjściem (słynny eksperyment Helbinga "faster is slower", rozbijający łuki zatorowe); 35 agentów.
6. **`scenarios/concert_stage.json`** – scena koncertowa / festiwal z dużą przeszkodą (scena) i dwoma szerokimi wyjściami bocznymi (górnym i dolnym); 60 agentów.

### Scenariusze geometrii korytarzy i przeszkód:
7. **`scenarios/classroom_evacuation.json`** – sala wykładowa z 10 rzędami ławek, wąskimi przejściami po obu stronach i wyjściem po lewej stronie; 40 agentów.
8. **`scenarios/corner_corridor.json`** – korytarz w kształcie litery L (zakręt 90 stopni). Wymaga omijania narożnika wewnętrznego; 30 agentów.
9. **`scenarios/t_junction.json`** – skrzyżowanie korytarzy typu T; dwa strumienie agentów z lewej i prawej strony łączą się w jeden korytarz wyjściowy; 40 agentów.
10. **`scenarios/hall_with_pillars.json`** – przestronny hol/stacja metra z regularną siatką filarów i szerokim wyjściem; 40 agentów.
11. **`scenarios/room_with_obstacles.json`** – pokój z dwoma symetrycznymi blokami przeszkód; 30 agentów.

---

## Dane rzeczywiste
- `crowdsim/data/scenarios/real_data_scenario.json` – scenariusz oparty o eksperyment Jülich (https://ped.fz-juelich.de/da/doku.php?id=corridor5).
- Narzędzia do przetwarzania danych w `crowdsim/data/`.

