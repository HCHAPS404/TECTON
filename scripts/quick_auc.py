"""AUC rápido del clasificador por config: 5 folds + stress lejano (13-24 m). Solo agregados."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from tecton.features import Features
from tecton.models import Models
from tecton.pipeline import fold_indices, recency_weights, with_external, with_oni
from tecton.schema import load_inputs

frames = load_inputs(Path("data/raw"), True)[0]
train = frames[0]
for path in sys.argv[1:]:
    config = with_oni(with_external(Path.cwd(), json.loads(Path(path).read_text()), set(train["DIVIPOLA"])), frames)
    splits = [(train.loc[tr], train.loc[va]) for tr, va in fold_indices(train, config["folds"])]
    splits.append((train[train.fecha <= "2020-09-01"], train[train.fecha.between("2021-10-01", "2022-09-01")]))
    aucs = []
    for tr, va in splits:
        fm = Features(**config["features"])
        x_tr = fm.fit_transform(tr)
        model = Models(config).fit_classifier_only(x_tr, tr.tiene_evento.to_numpy(), recency_weights(tr.fecha, config))
        aucs.append(roc_auc_score(va.tiene_evento, model.predict_proba_only(fm.transform(va))))
    objective = 0.5 * np.mean(aucs[:-1]) + 0.5 * aucs[-1]
    print(f"{Path(path).stem:30s} objetivo={objective:.4f} folds={np.mean(aucs[:-1]):.4f} lejano={aucs[-1]:.4f} {[round(a, 4) for a in aucs]}", flush=True)
