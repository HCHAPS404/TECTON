"""Búsqueda Optuna del clasificador con el protocolo temporal de TECTON. Solo imprime agregados.

Objetivo: AUC medio de los 5 folds + AUC del stress lejano (13-24 meses), con igual peso.
El AUC domina puntos_75; magnitud e intervalo se confirman después con el run completo.
Uso: uv run --frozen --with optuna python scripts/optuna_search.py --trials 40
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from tecton.features import Features
from tecton.models import Models
from tecton.pipeline import fold_indices
from tecton.schema import load_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, default=Path("configs/bagging.json"))
parser.add_argument("--trials", type=int, default=40)
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--out", type=Path, default=Path("configs/exp/optuna-best.json"))
parser.add_argument("--family", default="hist", choices=["hist", "lgbm"])
args = parser.parse_args()

base = json.loads(args.base.read_text())
train = load_inputs(Path("data/raw"), True)[0][0]
splits = [(train.loc[tr], train.loc[va]) for tr, va in fold_indices(train, base["folds"])]
splits.append((train[train.fecha <= "2020-09-01"], train[train.fecha.between("2021-10-01", "2022-09-01")]))
cache = {}


def features(smoothing):
    # Las features solo dependen del suavizado: se calculan una vez por valor.
    if smoothing not in cache:
        cache[smoothing] = []
        for tr, va in splits:
            model = Features(**{**base["features"], "smoothing": smoothing})
            cache[smoothing].append((model.fit_transform(tr), tr.tiene_evento.to_numpy(), model.transform(va), va.tiene_evento.to_numpy()))
    return cache[smoothing]


def objective(trial):
    smoothing = trial.suggest_categorical("smoothing", [5.0, 10.0, 20.0, 40.0])
    if args.family == "lgbm":
        model_params = {
            "family": "lgbm", "depth": -1,
            "iterations": trial.suggest_int("iterations", 200, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 600, log=True),
            "l2": trial.suggest_float("l2", 0.1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.3, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        config = {**base, "threads": args.threads, "model": model_params}
        return evaluate(trial, smoothing, config)
    model_params = {
        "family": "hist",
        "iterations": trial.suggest_int("iterations", 60, 400, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 400, log=True),
        "l2": trial.suggest_float("l2", 0.01, 20, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
    }
    config = {**base, "threads": args.threads, "model": model_params}
    return evaluate(trial, smoothing, config)


def evaluate(trial, smoothing, config):
    aucs = []
    for x_tr, y_tr, x_va, y_va in features(smoothing):
        model = Models(config)
        # Solo el clasificador: es lo que decide el AUC.
        model.fit_classifier_only(x_tr, y_tr)
        aucs.append(roc_auc_score(y_va, model.predict_proba_only(x_va)))
    trial.set_user_attr("fold_aucs", [round(a, 4) for a in aucs])
    return 0.5 * float(np.mean(aucs[:-1])) + 0.5 * aucs[-1]


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
# Punto de partida: la configuración vigente, para medir la mejora contra ella.
start = {"smoothing": base["features"]["smoothing"], "iterations": base["model"]["iterations"],
         "learning_rate": base["model"]["learning_rate"], "min_samples_leaf": base["model"].get("min_samples_leaf", 20),
         "l2": base["model"].get("l2", 2.0), "max_leaf_nodes": base["model"].get("max_leaf_nodes", 31),
         "max_features": base["model"].get("max_features", 1.0)}
if args.family == "lgbm":
    start["subsample"] = base["model"].get("subsample", 1.0)
else:
    start["depth"] = base["model"]["depth"]
study.enqueue_trial(start)


def report(study, trial):
    flag = " <- mejor" if study.best_trial.number == trial.number else ""
    print(f"trial {trial.number:3d} objetivo={trial.value:.4f} folds+lejano={trial.user_attrs['fold_aucs']}{flag}", flush=True)


study.optimize(objective, n_trials=args.trials, callbacks=[report])
best = study.best_trial
print("Mejor:", json.dumps(best.params), "objetivo", round(best.value, 4), "| referencia (trial 0)", round(study.trials[0].value, 4))
config = json.loads(json.dumps(base))
config["name"] = "optuna-best"
config["features"]["smoothing"] = best.params["smoothing"]
config["model"].update({k: v for k, v in best.params.items() if k != "smoothing"})
args.out.write_text(json.dumps(config, indent=2) + "\n")
print("Config escrita en", args.out)
