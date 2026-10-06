"""Comparación pareada de dos runs con bootstrap por bloques de mes. Solo imprime agregados."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tecton.metrics import score

parser = argparse.ArgumentParser()
parser.add_argument("run_a", help="Referencia")
parser.add_argument("run_b", help="Candidato")
parser.add_argument("--root", type=Path, default=Path.cwd())
parser.add_argument("--boot", type=int, default=1000)
args = parser.parse_args()

KEYS = ["DIVIPOLA", "fecha"]
COLS = ["prob", "point_log", "q10_log", "q90_log"]


def load(run_id):
    frame = pd.read_csv(args.root / "runs" / run_id / "oof.csv", dtype={"DIVIPOLA": "string"})
    return frame.sort_values(KEYS).reset_index(drop=True)


a, b = load(args.run_a), load(args.run_b)
if not a[KEYS].equals(b[KEYS]):
    raise SystemExit("Los OOF no comparten filas: no son comparables.")
event, people = a["tiene_evento"].to_numpy(), a["personas_desplazadas"].to_numpy()
pa, pb = [a[c].to_numpy() for c in COLS], [b[c].to_numpy() for c in COLS]
metrics = ["auc", "rmse_log", "winkler_log", "coverage_80", "puntos_75"]


def delta(index):
    ma = score(event[index], people[index], [v[index] for v in pa])
    mb = score(event[index], people[index], [v[index] for v in pb])
    return {m: mb[m] - ma[m] for m in metrics}


months = a["fecha"].to_numpy()
groups = {m: np.flatnonzero(months == m) for m in np.unique(months)}
keys = list(groups)
rng = np.random.default_rng(42)
point = delta(np.arange(len(a)))
samples = []
for _ in range(args.boot):
    chosen = rng.choice(len(keys), size=len(keys), replace=True)
    samples.append(delta(np.concatenate([groups[keys[i]] for i in chosen])))
samples = pd.DataFrame(samples)
print(f"B - A ({args.run_b} vs {args.run_a}); bootstrap por {len(keys)} meses, B={args.boot}")
for m in metrics:
    lo, hi = np.percentile(samples[m], [2.5, 97.5])
    better = (samples[m] < 0).mean() if m in ["rmse_log", "winkler_log"] else (samples[m] > 0).mean()
    print(f"  Δ{m:12s} {point[m]:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  P(mejora)={better:.2f}")
folds = []
for fold, part in a.groupby("fold"):
    index = part.index.to_numpy()
    folds.append({"fold": int(fold), **{m: round(v, 4) for m, v in delta(index).items()}})
print("Deltas por fold:", json.dumps(folds))
print(f"Folds con Δpuntos > 0: {sum(f['puntos_75'] > 0 for f in folds)}/{len(folds)}")
