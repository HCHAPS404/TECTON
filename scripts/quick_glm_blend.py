"""Logit econométrico (splines + depto x mes) solo y mezclado con LightGBM. Solo agregados."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from tecton.features import Features
from tecton.models import Models
from tecton.pipeline import fold_indices, with_external, with_oni
from tecton.schema import load_inputs

frames = load_inputs(Path("data/raw"), True)[0]
train = frames[0]
config = with_oni(with_external(Path.cwd(), json.loads(Path(sys.argv[1]).read_text()), set(train["DIVIPOLA"])), frames)
splits = [(train.loc[tr], train.loc[va]) for tr, va in fold_indices(train, config["folds"])]
splits.append((train[train.fecha <= "2020-09-01"], train[train.fecha.between("2021-10-01", "2022-09-01")]))
weights = [0.0, 0.15, 0.3, 0.5, 1.0]
results = {w: [] for w in weights}
for tr, va in splits:
    fm = Features(**config["features"])
    x_tr, x_va = fm.fit_transform(tr), fm.transform(va)
    for x in (x_tr, x_va):
        x["depto_mes"] = x["departamento"] + "_" + x["mes_cat"]
    cats = ["departamento", "mes_cat", "fase_enso", "depto_mes"]
    nums = [c for c in x_tr.columns if c not in cats + ["DIVIPOLA"]]
    glm = make_pipeline(
        ColumnTransformer([
            ("s", make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3)), nums),
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
        ]),
        LogisticRegression(C=0.05, max_iter=3000),
    )
    glm.fit(x_tr[nums + cats].fillna(0), tr.tiene_evento)
    p_glm = glm.predict_proba(x_va[nums + cats].fillna(0))[:, 1]
    gbm = Models(config).fit_classifier_only(x_tr.drop(columns="depto_mes"), tr.tiene_evento.to_numpy())
    p_gbm = gbm.predict_proba_only(x_va.drop(columns="depto_mes"))
    for w in weights:
        blend = (1 - w) * rankdata(p_gbm) + w * rankdata(p_glm)
        results[w].append(roc_auc_score(va.tiene_evento, blend))
for w, aucs in results.items():
    label = "LightGBM solo" if w == 0 else ("logit solo" if w == 1 else f"mezcla logit {w:.2f}")
    print(f"{label:22s} folds={np.mean(aucs[:-1]):.4f} lejano={aucs[-1]:.4f} objetivo={0.5 * np.mean(aucs[:-1]) + 0.5 * aucs[-1]:.4f}", flush=True)
