"""Recalibración temporal del punto y del intervalo en escala log1p.

Se ajusta sobre los últimos meses del entrenamiento de cada fold con un modelo entrenado
solo con los meses anteriores, igual que la inferencia futura con historial congelado.
Nunca usa la validación ni la prueba.
"""

import numpy as np

SCALES = np.linspace(0.5, 1.6, 23)
SHIFTS = np.linspace(-1.5, 1.5, 61)


def _best(loss_fn, bound):
    best, params = np.inf, (1.0, 0.0)
    for scale in SCALES:
        candidates = np.maximum(0, scale * bound[:, None] + SHIFTS[None, :])
        losses = loss_fn(candidates).mean(axis=0)
        i = int(np.argmin(losses))
        if losses[i] < best:
            best, params = float(losses[i]), (float(scale), float(SHIFTS[i]))
    return params


def fit(z, prob, point, low, high, penalty=10.0):
    """Parámetros que minimizan RMSE (punto) y Winkler alfa 0.2 (cada extremo por separado)."""
    z = np.asarray(z, dtype=float)
    design = np.column_stack([point, np.ones_like(point)])
    slope, intercept = np.linalg.lstsq(design, z, rcond=None)[0]
    # Winkler = (hi - lo) + p(lo - z)+ + p(z - hi)+ se separa en un término por extremo.
    high_loss = lambda h: h + penalty * np.maximum(z[:, None] - h, 0)  # noqa: E731
    # Con ~90% de ceros, el q90 real es 0 donde el riesgo es bajo: umbral de probabilidad calibrado.
    best = (np.inf, 0.0, (1.0, 0.0))
    for threshold in np.unique(np.quantile(prob, np.linspace(0, 0.95, 20))):
        gate = prob >= threshold
        params = _best(lambda h: np.where(gate[:, None], high_loss(h), high_loss(np.zeros_like(h))), high)
        candidate = np.where(gate, np.maximum(0, params[0] * high + params[1]), 0)
        loss = float(high_loss(candidate[:, None]).mean())
        if loss < best[0]:
            best = (loss, float(threshold), params)
    low_params = _best(lambda lo: -lo + penalty * np.maximum(lo - z[:, None], 0), low)
    return {"point": (float(np.clip(slope, 0.3, 1.7)), float(intercept)), "low": low_params,
            "high": best[2], "high_threshold": best[1]}


def apply(params, predictions):
    prob, point, low, high = predictions
    a, b = params["point"]
    point = np.maximum(0, a * point + b)
    low = np.maximum(0, params["low"][0] * low + params["low"][1])
    high = np.maximum(0, params["high"][0] * high + params["high"][1])
    high = np.where(prob >= params.get("high_threshold", 0.0), high, 0.0)
    return prob, point, np.minimum(low, high), np.maximum(low, high)
