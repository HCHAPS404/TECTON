"""Deriva entre entrenamiento y prueba: validación adversaria y PSI por covariable. Solo agregados."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from tecton.schema import NUMERIC, load_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
args = parser.parse_args()
frames, _, _, _ = load_inputs(args.data_dir, True)
train, test = frames[0], pd.concat(frames[1:], ignore_index=True)


def psi(reference, current, bins=10):
    edges = np.unique(np.nanquantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref = np.histogram(np.clip(reference, edges[0], edges[-1]), edges)[0] / len(reference)
    cur = np.histogram(np.clip(current, edges[0], edges[-1]), edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


print("PSI entrenamiento -> prueba (>0.25 deriva fuerte):")
for col in NUMERIC:
    print(f"  {col:26s} {psi(train[col].dropna().to_numpy(), test[col].dropna().to_numpy()):.3f}")
for name, part in [("pública", frames[1]), ("privada", frames[2])]:
    shares = part["fase_enso"].value_counts(normalize=True).round(3).to_dict()
    print(f"Fase ENSO {name}: {shares}")
print(f"Fase ENSO entrenamiento: {train['fase_enso'].value_counts(normalize=True).round(3).to_dict()}")
both = pd.concat([train[NUMERIC].assign(t=0), test[NUMERIC].assign(t=1)], ignore_index=True)
# Sin ONI ni fecha: medir si las covariables municipales delatan el período.
cols = [c for c in NUMERIC if c != "ONI"]
pred = cross_val_predict(HistGradientBoostingClassifier(max_iter=100), both[cols], both["t"], cv=5, method="predict_proba")[:, 1]
print(f"AUC adversario (covariables municipales, sin ONI): {roc_auc_score(both['t'], pred):.3f} (0.5 = sin deriva)")
