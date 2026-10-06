"""GARCH(1,1) de panel sobre log1p(personas_desplazadas) por municipio.

    e_it      = y_it - mu_it
    sigma2_it = (1 - alpha - beta) * v_it + alpha * e_i,t-1^2 + beta * sigma2_i,t-1

mu_it y v_it son media y varianza expandidas con meses estrictamente anteriores, encogidas
hacia el panel (variance targeting). alpha y beta son comunes a los municipios y se estiman
por cuasi-verosimilitud gaussiana solo con meses <= corte. Después del corte no hay residuos
observados: el pronóstico a h meses es v + (alpha + beta)^(h-1) * (sigma2_c+1 - v), igual en
la prueba pública y en la privada.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SHRINK_MONTHS = 6.0
WARMUP = 6
FLOOR = 1e-3


def _panel(train, cutoff):
    rows = train[train["fecha"] <= cutoff]
    y = np.log1p(rows["personas_desplazadas"].to_numpy(float))
    table = pd.DataFrame({"DIVIPOLA": rows["DIVIPOLA"].astype(str).str.zfill(5).to_numpy(),
                          "fecha": pd.to_datetime(rows["fecha"]).to_numpy(), "y": y})
    wide = table.pivot_table(index="DIVIPOLA", columns="fecha", values="y", aggfunc="sum").sort_index(axis=1)
    return wide.fillna(0.0)


def _priors(values):
    """Media y varianza con columnas < t, encogidas hacia el panel. Columna T = todo el período."""
    n, t = values.shape
    zeros = np.zeros((n, 1))
    csum = np.hstack([zeros, np.cumsum(values, axis=1)])
    csq = np.hstack([zeros, np.cumsum(values ** 2, axis=1)])
    k = np.arange(t + 1, dtype=float)
    pooled_sum, pooled_sq = csum.sum(axis=0), csq.sum(axis=0)
    pooled_n = np.maximum(k * n, 1.0)
    pooled_mu = pooled_sum / pooled_n
    pooled_var = np.maximum(pooled_sq / pooled_n - pooled_mu ** 2, FLOOR)
    pooled_var[0] = 1.0
    weight = k / (k + SHRINK_MONTHS)
    own_mu = csum / np.maximum(k, 1.0)
    own_var = np.maximum(csq / np.maximum(k, 1.0) - own_mu ** 2, FLOOR)
    mu = weight * own_mu + (1 - weight) * pooled_mu
    var = np.maximum(weight * own_var + (1 - weight) * pooled_var, FLOOR)
    return mu, var


def _filter(values, mu, var, alpha, beta):
    """sigma2 de una columna más que values: la última es el paso fuera de muestra."""
    n, t = values.shape
    sigma2 = np.empty((n, t + 1))
    sigma2[:, 0] = var[:, 0]
    omega = 1.0 - alpha - beta
    for j in range(1, t + 1):
        shock = (values[:, j - 1] - mu[:, j - 1]) ** 2
        sigma2[:, j] = omega * var[:, j] + alpha * shock + beta * sigma2[:, j - 1]
    return np.maximum(sigma2, FLOOR)


def _nll(params, values, mu, var):
    alpha, beta = params
    if alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e18
    sigma2 = _filter(values, mu, var, alpha, beta)[:, WARMUP:-1]
    resid = values[:, WARMUP:] - mu[:, WARMUP:-1]
    return 0.5 * float(np.mean(np.log(sigma2) + resid ** 2 / sigma2))


class PanelGarch:
    def fit(self, train, cutoff, params=None):
        self.cutoff = pd.Timestamp(cutoff)
        wide = _panel(train, self.cutoff)
        self.codes = list(wide.index)
        self.dates = list(pd.to_datetime(wide.columns))
        values = wide.to_numpy(float)
        mu, var = _priors(values)
        if params is not None:
            self.alpha, self.beta = (float(v) for v in params)
        elif values.shape[1] <= WARMUP + 2:
            self.alpha, self.beta = 0.0, 0.0
        else:
            result = minimize(_nll, x0=np.array([0.1, 0.6]), args=(values, mu, var), method="L-BFGS-B",
                              bounds=[(0.0, 0.6), (0.0, 0.98)])
            self.alpha, self.beta = (float(v) for v in result.x)
            if self.alpha + self.beta >= 0.999:
                scale = 0.998 / (self.alpha + self.beta)
                self.alpha, self.beta = self.alpha * scale, self.beta * scale
        self.mu, self.var = mu, var
        self.sigma2 = _filter(values, mu, var, self.alpha, self.beta)
        self.global_mu = float(mu[:, -1].mean())
        self.global_var = float(var[:, -1].mean())
        return self

    def transform(self, frame):
        """Varianza condicional para cada fila: un paso dentro del corte, h pasos después."""
        codes = frame["DIVIPOLA"].astype(str).str.zfill(5).to_numpy()
        dates = pd.to_datetime(frame["fecha"]).to_numpy()
        row = pd.Series(np.arange(len(self.codes)), index=self.codes).reindex(codes).to_numpy()
        known = ~np.isnan(row)
        row = np.where(known, row, 0).astype(int)
        col_map = pd.Series(np.arange(len(self.dates)), index=pd.DatetimeIndex(self.dates))
        col = col_map.reindex(pd.DatetimeIndex(dates)).to_numpy()
        inside = ~np.isnan(col)
        col = np.where(inside, col, 0).astype(int)
        last = len(self.dates)
        cutoff = pd.Timestamp(self.cutoff)
        stamps = pd.DatetimeIndex(dates)
        horizon = np.maximum((stamps.year - cutoff.year) * 12 + stamps.month - cutoff.month, 1).to_numpy()
        persistence = self.alpha + self.beta
        v_end = self.var[row, last]
        s_next = self.sigma2[row, last]
        forecast = v_end + persistence ** (horizon - 1) * (s_next - v_end)
        sigma2 = np.where(inside, self.sigma2[row, col], forecast)
        mu = np.where(inside, self.mu[row, col], self.mu[row, last])
        var = np.where(inside, self.var[row, col], v_end)
        sigma2 = np.where(known, sigma2, self.global_var)
        mu = np.where(known, mu, self.global_mu)
        var = np.where(known, var, self.global_var)
        return pd.DataFrame({
            "garch_sigma": np.sqrt(sigma2),
            "garch_media": mu,
            "garch_sigma_incondicional": np.sqrt(var),
            "garch_ratio": sigma2 / var,
        }, index=frame.index)
