import numpy as np
from sklearn.metrics import roc_auc_score


def score(event, people, predictions):
    prob, point, low, high = predictions
    z = np.log1p(np.asarray(people, dtype=float))
    if len(np.unique(event)) != 2:
        raise ValueError("AUC no está definido: validación contiene una sola clase.")
    winkler = high - low + 10 * np.maximum(low - z, 0) + 10 * np.maximum(z - high, 0)
    result = {
        "auc": float(roc_auc_score(event, prob)),
        "rmse_log": float(np.sqrt(np.mean((z - point) ** 2))),
        "winkler_log": float(np.mean(winkler)),
        "coverage_80": float(np.mean((low <= z) & (z <= high))),
        "interval_width_log": float(np.mean(high - low)),
    }
    result["puntos_75"] = points(result, z)
    return result


def points(metrics, z):
    """Parte automática de la nota (máx. 75), según el notebook base PNUD (T&C 6.3).

    La referencia nula predice prob 0.5 y cero personas con intervalo [0, 0]:
    RMSE nulo = sqrt(mean(z^2)) y Winkler nulo = mean(10 z).
    """
    rmse_null = float(np.sqrt(np.mean(z ** 2)))
    winkler_null = float(np.mean(10 * z))
    s_auc = max(0.0, (metrics["auc"] - 0.5) / 0.5)
    s_rmse = min(1.0, max(0.0, 1 - metrics["rmse_log"] / rmse_null)) if rmse_null > 0 else 0.0
    s_wink = min(1.0, max(0.0, 1 - metrics["winkler_log"] / winkler_null)) if winkler_null > 0 else 0.0
    return float(100 * (0.50 * s_auc + 0.20 * s_rmse + 0.05 * s_wink))
