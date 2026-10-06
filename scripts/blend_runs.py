"""Mezcla de runs completos con pesos iguales, equivalente a la config `ensemble` del pipeline.

Promedia prob y log1p(punto, q10, q90) de los OOF (para medir) y de predicciones.csv (para entregar).
Uso: uv run --frozen python scripts/blend_runs.py REF_RUN RUN_A RUN_B [RUN_C ...]
REF_RUN es la referencia de la comparación (champion o último envío). Solo imprime agregados.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tecton.metrics import score
from tecton.pipeline import verify_run_artifacts
from tecton.schema import KEYS, OUTPUT, load_inputs, sha256, validate_predictions

root = Path.cwd()
reference, members = sys.argv[1], sys.argv[2:]
COLS = ["prob", "point_log", "q10_log", "q90_log"]


def oof(run_id):
    return pd.read_csv(root / "runs" / run_id / "oof.csv", dtype={"DIVIPOLA": "string"}).sort_values(KEYS).reset_index(drop=True)


def fold_metrics(frame, vectors):
    folds = [score(g.tiene_evento, g.personas_desplazadas, [v[g.index] for v in vectors]) for _, g in frame.groupby("fold")]
    return {k: float(np.mean([f[k] for f in folds])) for k in ["puntos_75", "auc", "rmse_log", "winkler_log", "coverage_80"]}


frames = [oof(r) for r in members]
base = oof(reference)
for f in frames:
    if not f[KEYS].equals(base[KEYS]):
        raise SystemExit("Los OOF no comparten filas.")
blend = [np.mean([f[c].to_numpy() for f in frames], axis=0) for c in COLS]
print("Referencia", reference, {k: round(v, 4) for k, v in fold_metrics(base, [base[c].to_numpy() for c in COLS]).items()})
for run_id, f in zip(members, frames):
    print("Miembro   ", run_id, {k: round(v, 4) for k, v in fold_metrics(f, [f[c].to_numpy() for c in COLS]).items()})
print("Mezcla    ", {k: round(v, 4) for k, v in fold_metrics(base, blend).items()})

# Predicciones de entrega: mismo promedio en log1p.
preds = []
for run_id in members:
    run = root / "runs" / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    verify_run_artifacts(run, manifest)
    preds.append(pd.read_csv(run / "predicciones.csv", dtype={"DIVIPOLA": "string"}))
for p in preds[1:]:
    if not p[KEYS].equals(preds[0][KEYS]):
        raise SystemExit("Las predicciones no comparten llaves y orden.")
out = preds[0][KEYS].copy()
out["prob_evento"] = np.mean([p["prob_evento"].to_numpy() for p in preds], axis=0)
for col in OUTPUT[3:]:
    out[col] = np.expm1(np.mean([np.log1p(p[col].to_numpy()) for p in preds], axis=0))
low, high = out["personas_desplazadas_q10"].to_numpy(), out["personas_desplazadas_q90"].to_numpy()
out["personas_desplazadas_q10"], out["personas_desplazadas_q90"] = np.minimum(low, high), np.maximum(low, high)
test = load_inputs(root / "data/raw", True)[0]
validate_predictions(out, pd.concat([f[KEYS] for f in test[1:]], ignore_index=True))
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
target = root / "colab" / f"mezcla-{stamp}"
target.mkdir(parents=True)
out.to_csv(target / f"predicciones-mezcla-{stamp}.csv", index=False)
(target / "mezcla.json").write_text(json.dumps({"members": members, "weights": "iguales", "reference": reference,
    "sha256": sha256(target / f"predicciones-mezcla-{stamp}.csv")}, indent=2))
print("CSV de mezcla validado:", target / f"predicciones-mezcla-{stamp}.csv")
