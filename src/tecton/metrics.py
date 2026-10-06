import numpy as np
from sklearn.metrics import roc_auc_score


def score(event, people, predictions):
    prob, point, low, high = predictions
    z = np.log1p(np.asarray(people, dtype=float))
    if len(np.unique(event)) != 2:
        raise ValueError("AUC no está definido: validación contiene una sola clase.")
    winkler = high - low + 10 * np.maximum(low - z, 0) + 10 * np.maximum(z - high, 0)
    return {
        "auc": float(roc_auc_score(event, prob)),
        "rmse_log": float(np.sqrt(np.mean((z - point) ** 2))),
        "winkler_log": float(np.mean(winkler)),
        "coverage_80": float(np.mean((low <= z) & (z <= high))),
        "interval_width_log": float(np.mean(high - low)),
    }
