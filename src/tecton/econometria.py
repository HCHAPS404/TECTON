"""Panel Mundlak + GARCH(1,1) de varianza + CatBoost, alineado a los cortes del reto.

1. Entrena solo con meses etiquetados hasta el corte del fold (como máximo 2022-09).
2. La prueba pública se predice con esa historia congelada.
3. La prueba privada usa el mismo coeficiente y alarga la historia con las predicciones
   públicas, no con reportes reales posteriores a 2022-09.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize

from tecton.features import Features
from tecton.fuentes_publicas import CLIMATE, load_dane, load_ungrd
from tecton.garch import PanelGarch

OFFICIAL_COUNTS = [
    "n_inundacion", "n_movimiento_masa", "n_vendaval", "n_creciente_subita",
    "n_avenida_torrencial", "n_otros", "viv_destruidas", "viv_averiadas",
]
SLOW = [
    "densidad_poblacional", "ingresos_tributarios", "inversion_gestion_riesgo",
    "nbi_pct", "elevacion_media", "pendiente_media", "pct_pendiente_alta",
]
HISTORY = [
    "history_event_rate", "history_department_rate", "history_seasonal_rate",
    "history_mean_log_people", "history_positive_people_rate",
]
PUBLIC_END = pd.Timestamp("2024-04-01")
PRIVATE_START = pd.Timestamp("2024-05-01")
TRAIN_CAP = pd.Timestamp("2022-09-01")


def prior_means(frame, monthly, columns):
    """Media de meses estrictamente anteriores. monthly ya debe estar cortada al fold."""
    names = [f"lag_{c}" for c in columns]
    result = pd.DataFrame(0.0, index=np.arange(len(frame)), columns=names)
    if monthly is None or monthly.empty or not columns:
        result.index = frame.index
        return result
    work = monthly.dropna(subset=["DIVIPOLA", "fecha"]).copy()
    work["DIVIPOLA"] = work["DIVIPOLA"].astype(str).str.zfill(5)
    work["fecha"] = pd.to_datetime(work["fecha"])
    work = work.groupby(["DIVIPOLA", "fecha"], as_index=False)[columns].sum().sort_values(["DIVIPOLA", "fecha"])
    grouped = work.groupby("DIVIPOLA", sort=False)
    work[columns] = grouped[columns].cumsum()
    work["k"] = grouped.cumcount() + 1
    query = frame[["DIVIPOLA", "fecha"]].copy()
    query["DIVIPOLA"] = query["DIVIPOLA"].astype(str).str.zfill(5)
    query["fecha"] = pd.to_datetime(query["fecha"])
    query["_ord"] = np.arange(len(query))
    query["fecha_key"] = query["fecha"] - pd.Timedelta(days=1)
    states = {code: part.sort_values("fecha") for code, part in work.groupby("DIVIPOLA", sort=False)}
    pieces = []
    for code, part in query.groupby("DIVIPOLA", sort=False):
        state = states.get(code)
        ordered = part.sort_values("fecha_key")
        if state is None or state.empty:
            hit = ordered[["_ord"]].copy()
            for name in names:
                hit[name] = 0.0
        else:
            hit = pd.merge_asof(
                ordered, state, left_on="fecha_key", right_on="fecha", direction="backward",
            )
            for column, name in zip(columns, names, strict=True):
                hit[name] = hit[column] / hit["k"]
            hit[names] = hit[names].fillna(0.0)
        pieces.append(hit[["_ord", *names]])
    stacked = pd.concat(pieces, ignore_index=True).sort_values("_ord")
    result = stacked[names].reset_index(drop=True)
    result.index = frame.index
    return result


class HurdleMundlak:
    """Tres ecuaciones: cloglog del evento, valla de log1p y cuantiles 0.10/0.90."""

    def __init__(self, l2=1.0):
        self.l2 = float(l2)

    def fit(self, x, event, log_people):
        self.columns = list(x.columns)
        matrix = self._scale_fit(x.to_numpy(float))
        event = np.asarray(event, float)
        log_people = np.asarray(log_people, float)
        if len(np.unique(event)) < 2 or not ((log_people > 0).any() and (log_people == 0).any()):
            raise ValueError("El fold no tiene ambas clases del evento y de la magnitud.")
        self.event_coef = self._cloglog(matrix, event)
        positive = log_people > 0
        self.positive_coef = self._cloglog(matrix, positive.astype(float))
        self.magnitude_coef = self._ridge(matrix[positive], log_people[positive])
        # Cuantiles de la parte positiva. El cero se mezcla al predecir: con 92% de ceros,
        # el percentil 10 condicional es 0 y el 90 solo se abre donde hay masa positiva.
        self.low_coef = self._quantile(matrix[positive], log_people[positive], 0.1)
        self.high_coef = self._quantile(matrix[positive], log_people[positive], 0.9)
        return self

    def predict(self, x):
        matrix = self._scale_apply(x[self.columns].to_numpy(float))
        prob = self._cloglog_prob(matrix @ self.event_coef)
        p_pos = self._cloglog_prob(matrix @ self.positive_coef)
        magnitude = np.clip(matrix @ self.magnitude_coef, 0, 12)
        point = np.clip(p_pos * magnitude, 0, 12)
        low_pos = np.clip(matrix @ self.low_coef, 0, 12)
        high_pos = np.clip(matrix @ self.high_coef, 0, 12)
        low_pos, high_pos = np.minimum(low_pos, high_pos), np.maximum(low_pos, high_pos)
        low = self._mixture_quantile(p_pos, low_pos, high_pos, 0.1)
        high = self._mixture_quantile(p_pos, low_pos, high_pos, 0.9)
        return prob, point, low, high

    @staticmethod
    def _mixture_quantile(p_pos, low_pos, high_pos, tau):
        """Cuantil de la mezcla cero + parte positiva. Si P(Y>0) no alcanza tau, el cuantil es 0."""
        out = np.zeros(len(p_pos))
        need = p_pos > (1 - tau)
        alpha = (tau - (1 - p_pos[need])) / p_pos[need]
        below = alpha < 0.1
        above = alpha > 0.9
        mid = ~below & ~above
        out_need = np.empty(need.sum())
        out_need[below] = low_pos[need][below] * (alpha[below] / 0.1)
        out_need[above] = high_pos[need][above]
        out_need[mid] = low_pos[need][mid] + (high_pos[need][mid] - low_pos[need][mid]) * ((alpha[mid] - 0.1) / 0.8)
        out[need] = out_need
        return np.clip(out, 0, 12)

    def _scale_fit(self, matrix):
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        scaled = (matrix - self.mean) / self.scale
        return np.column_stack([np.ones(len(matrix)), scaled])

    def _scale_apply(self, matrix):
        scaled = (matrix - self.mean) / self.scale
        return np.column_stack([np.ones(len(matrix)), scaled])

    def _cloglog(self, matrix, y):
        l2 = self.l2

        def objective(beta):
            eta = np.clip(matrix @ beta, -20, 20)
            exposure = np.exp(eta)
            # P = 1 - exp(-exp(eta)); estable vía expm1.
            prob = -np.expm1(-exposure)
            prob = np.clip(prob, 1e-12, 1 - 1e-12)
            nll = -(y * np.log(prob) + (1 - y) * (-exposure)).sum()
            nll += 0.5 * l2 * np.dot(beta[1:], beta[1:])
            grad_eta = -y * (exposure * (1 - prob) / prob) + (1 - y) * exposure
            grad = matrix.T @ grad_eta
            grad[1:] += l2 * beta[1:]
            return nll, grad

        start = np.zeros(matrix.shape[1])
        start[0] = np.log(-np.log(max(1e-4, 1 - y.mean())))
        result = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        if not np.isfinite(result.fun):
            raise ValueError("El cloglog no convergió.")
        return result.x

    @staticmethod
    def _cloglog_prob(eta):
        return np.clip(-np.expm1(-np.exp(np.clip(eta, -20, 20))), 1e-6, 1 - 1e-6)

    def _ridge(self, matrix, y):
        penalty = self.l2 * np.eye(matrix.shape[1])
        penalty[0, 0] = 0.0
        return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ y)

    def _quantile(self, matrix, y, tau, rounds=40):
        beta = np.zeros(matrix.shape[1])
        beta[0] = float(np.quantile(y, tau))
        identity = self.l2 * np.eye(matrix.shape[1])
        identity[0, 0] = 0.0
        for _ in range(rounds):
            resid = y - matrix @ beta
            weights = np.where(resid >= 0, tau, 1 - tau) / np.maximum(np.abs(resid), 1e-3)
            weighted = matrix * weights[:, None]
            beta = np.linalg.solve(matrix.T @ weighted + identity, weighted.T @ y)
        return beta


def _take(frame, table, keys):
    """Une sin perder el orden de frame. table trae una fila por llave."""
    base = frame[keys].copy()
    base["_ord"] = np.arange(len(base))
    merged = base.merge(table, on=keys, how="left").sort_values("_ord")
    values = merged.drop(columns=[*keys, "_ord"])
    values.index = frame.index
    return values


def _mundlak(frame, columns):
    means = frame.groupby("DIVIPOLA")[columns].mean()
    means.columns = [f"mundlak_{c}" for c in columns]
    taken = _take(frame, means.reset_index(), ["DIVIPOLA"])
    return taken.fillna(means.mean())


def _dane_column(frame, dane, mundlak_table):
    if dane is None or dane.empty:
        level = pd.Series(0.0, index=frame.index)
    else:
        keys = pd.DataFrame({
            "DIVIPOLA": frame["DIVIPOLA"].astype(str).str.zfill(5).to_numpy(),
            "anio": frame["anio"].astype(int).clip(upper=2022).to_numpy(),
        }, index=frame.index)
        level = _take(keys, dane, ["DIVIPOLA", "anio"])["poblacion"]
        filled = level.median() if level.notna().any() else 0.0
        level = np.log1p(level.fillna(filled).astype(float))
        level = pd.Series(level, index=frame.index)
    if mundlak_table is None:
        between = level.groupby(frame["DIVIPOLA"].to_numpy()).transform("mean")
        between.index = frame.index
    else:
        between = _take(frame, mundlak_table, ["DIVIPOLA"])["dane_log_poblacion"]
        fallback = float(mundlak_table["dane_log_poblacion"].mean()) if len(mundlak_table) else 0.0
        between = between.fillna(fallback)
    return pd.DataFrame({"dane_log_poblacion": level, "mundlak_dane_log_poblacion": between}, index=frame.index)


def design(frame, x_features, dane, ungrd, cutoff, official_monthly, mundlak_table=None, dane_between=None,
           garch=None):
    """Arma X. Conteos oficiales y UNGRD pública entran solo con meses anteriores al corte."""
    base = pd.DataFrame(index=frame.index)
    base["ONI"] = x_features["ONI"].to_numpy()
    base["oni_pendiente"] = x_features["oni_pendiente"].to_numpy()
    base["oni_mes"] = x_features["ONI"].to_numpy() * x_features["mes_sin"].to_numpy()
    base["mes_sin"] = x_features["mes_sin"].to_numpy()
    base["mes_cos"] = x_features["mes_cos"].to_numpy()
    for column in SLOW:
        base[column] = x_features[column].to_numpy()
    for column in HISTORY:
        base[column] = x_features[column].to_numpy()
    if mundlak_table is None:
        slow = _mundlak(frame, [c for c in SLOW if c in frame.columns])
    else:
        slow = _take(frame, mundlak_table, ["DIVIPOLA"]).fillna(mundlak_table.mean(numeric_only=True))
    if official_monthly is None or official_monthly.empty:
        official = pd.DataFrame(columns=["DIVIPOLA", "fecha", *OFFICIAL_COUNTS])
    else:
        official = official_monthly[official_monthly["fecha"] <= cutoff]
    if ungrd is None or ungrd.empty:
        public = ungrd
    else:
        public = ungrd[ungrd["fecha"] <= min(pd.Timestamp(cutoff), TRAIN_CAP)]
    oficial_lag = prior_means(frame, official, OFFICIAL_COUNTS).add_prefix("oficial_")
    publico_lag = prior_means(frame, public, list(CLIMATE)).add_prefix("publico_")
    dane_cols = _dane_column(frame, dane, dane_between)
    blocks = [base, slow, dane_cols, oficial_lag, publico_lag]
    if garch is not None:
        blocks.append(garch.transform(frame))
    matrix = pd.concat(blocks, axis=1).astype(float).fillna(0.0)
    matrix["oni_historia"] = matrix["ONI"] * matrix["history_event_rate"]
    matrix["historia_pendiente"] = matrix["history_event_rate"] * matrix["pendiente_media"]
    if garch is not None:
        matrix["garch_oni"] = matrix["garch_sigma"] * matrix["ONI"]
    return matrix


def _categories(frame, x_features):
    return pd.DataFrame({
        "DIVIPOLA": frame["DIVIPOLA"].astype(str).str.zfill(5).to_numpy(),
        "departamento": x_features["departamento"].astype(str).to_numpy(),
        "mes_cat": x_features["mes_cat"].astype(str).to_numpy(),
    })


class PanelBoost:
    """CatBoost sobre el diseño temporal. DIVIPOLA entra como categoría, no como número."""

    def __init__(self, iterations=40, learning_rate=0.06, depth=6, seed=42, threads=2):
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.depth = int(depth)
        self.seed = int(seed)
        self.threads = int(threads)

    def fit(self, numeric, cats, event, log_people):
        self.num_columns = list(numeric.columns)
        self.cat_columns = list(cats.columns)
        self.columns = self.num_columns + self.cat_columns
        event = np.asarray(event, int)
        log_people = np.asarray(log_people, float)
        if len(np.unique(event)) < 2:
            raise ValueError("El fold no tiene ambas clases del evento.")
        frame = self._frame(numeric, cats)
        common = dict(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            random_seed=self.seed, thread_count=self.threads, verbose=False,
            allow_writing_files=False, cat_features=self.cat_columns,
        )
        self.classifier = CatBoostClassifier(loss_function="Logloss", **common)
        self.point = CatBoostRegressor(loss_function="RMSE", **common)
        self.low = CatBoostRegressor(loss_function="Quantile:alpha=0.1", leaf_estimation_method="Gradient", **common)
        self.high = CatBoostRegressor(loss_function="Quantile:alpha=0.9", leaf_estimation_method="Gradient", **common)
        self.classifier.fit(frame, event)
        self.point.fit(frame, log_people)
        self.low.fit(frame, log_people)
        self.high.fit(frame, log_people)
        return self

    def predict(self, numeric, cats):
        frame = self._frame(numeric, cats)
        prob = np.clip(self.classifier.predict_proba(frame)[:, 1], 1e-6, 1 - 1e-6)
        point = np.clip(self.point.predict(frame), 0, 12)
        low = np.clip(self.low.predict(frame), 0, 12)
        high = np.clip(self.high.predict(frame), 0, 12)
        low, high = np.minimum(low, high), np.maximum(low, high)
        return prob, point, low, high

    def _frame(self, numeric, cats):
        frame = numeric[self.num_columns].copy()
        for column in self.cat_columns:
            frame[column] = cats[column].astype(str).to_numpy()
        return frame


def _store_mundlak(frame):
    table = frame.groupby("DIVIPOLA")[SLOW].mean()
    table.columns = [f"mundlak_{c}" for c in SLOW]
    return table.reset_index()


def _store_dane_between(frame, dane):
    level = _dane_column(frame, dane, None)["dane_log_poblacion"]
    table = pd.DataFrame({"DIVIPOLA": frame["DIVIPOLA"].to_numpy(), "dane_log_poblacion": level.to_numpy()})
    return table.groupby("DIVIPOLA", as_index=False)["dane_log_poblacion"].mean()


def _fit_block(train, config, dane, ungrd):
    features = Features(**config["features"])
    x_features = features.fit_transform(train)
    cutoff = train["fecha"].max()
    official = train.loc[train["fecha"] <= cutoff, ["DIVIPOLA", "fecha", *OFFICIAL_COUNTS]].copy()
    mundlak = _store_mundlak(train)
    dane_between = _store_dane_between(train, dane)
    garch = PanelGarch().fit(train, cutoff) if config["model"].get("garch", False) else None
    if garch is not None:
        print(f"econ GARCH(1,1) corte {cutoff.date()}: alpha={garch.alpha:.3f} beta={garch.beta:.3f}", flush=True)
    matrix = design(train, x_features, dane, ungrd, cutoff, official, garch=garch)
    cats = _categories(train, x_features)
    params = config["model"]
    model = PanelBoost(
        iterations=params.get("iterations", 40), learning_rate=params.get("learning_rate", 0.06),
        depth=params.get("depth", 6), seed=config.get("seed", 42), threads=config.get("threads", 2),
    ).fit(matrix, cats, train["tiene_evento"].to_numpy(int), np.log1p(train["personas_desplazadas"].to_numpy(float)))
    state = {
        "features": features, "model": model, "mundlak": mundlak, "dane_between": dane_between,
        "cutoff": cutoff, "official": official, "garch": garch,
    }
    return state


def _predict_block(state, frame, dane, ungrd, x_features=None):
    if x_features is None:
        x_features = state["features"].transform(frame)
    matrix = design(
        frame, x_features, dane, ungrd, state["cutoff"], state["official"], state["mundlak"], state["dane_between"],
        state["garch"],
    )
    return state["model"].predict(matrix, _categories(frame, x_features))


def fit_predict_econ(train, test, config, root=None):
    root = Path(root) if root is not None else None
    dane = load_dane(root) if root is not None else pd.DataFrame(columns=["DIVIPOLA", "anio", "poblacion"])
    ungrd = load_ungrd(root) if root is not None else pd.DataFrame(columns=["DIVIPOLA", "fecha", *CLIMATE])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    state = _fit_block(train, config, dane, ungrd)
    public = test["fecha"] <= PUBLIC_END
    private = test["fecha"] >= PRIVATE_START
    staged = bool(public.any() and private.any() and test["fecha"].min() > state["cutoff"])
    if not staged:
        pred = _predict_block(state, test, dane, ungrd)
        return pred, state["features"], state["model"], list(state["model"].columns)

    codes = set(train["DIVIPOLA"].astype(str).str.zfill(5))
    dane_hit = len(codes & set(dane["DIVIPOLA"].astype(str))) / len(codes) if len(dane) else 0.0
    ungrd_hit = len(codes & set(ungrd["DIVIPOLA"].astype(str))) / len(codes) if len(ungrd) else 0.0
    print(
        f"econ etapa entrenamiento: {train['fecha'].min().date()} a {state['cutoff'].date()} ({len(train)} filas); cobertura DIVIPOLA DANE {dane_hit:.3f}, UNGRD pública {ungrd_hit:.3f}",
        flush=True,
    )
    public_frame = test.loc[public].reset_index(drop=True)
    private_frame = test.loc[private].reset_index(drop=True)
    public_pred = _predict_block(state, public_frame, dane, ungrd)
    print(
        f"econ etapa prueba pública: {public_frame['fecha'].min().date()} a {public_frame['fecha'].max().date()} ({len(public_frame)} filas), historia congelada en {state['cutoff'].date()}",
        flush=True,
    )
    pseudo = public_frame.copy()
    pseudo["tiene_evento"] = public_pred[0]
    pseudo["personas_desplazadas"] = np.expm1(public_pred[1])
    for column in OFFICIAL_COUNTS:
        if column not in pseudo.columns:
            pseudo[column] = 0.0
    extended = pd.concat([train, pseudo], ignore_index=True)
    history_model = Features(**config["features"])
    history_model.fit_transform(extended)
    private_x = state["features"].transform(private_frame)
    updated = history_model.transform(private_frame)
    for column in HISTORY:
        private_x[column] = updated[column].to_numpy()
    private_pred = _predict_block(state, private_frame, dane, ungrd, private_x)
    print(
        f"econ etapa prueba privada: {private_frame['fecha'].min().date()} a {private_frame['fecha'].max().date()} ({len(private_frame)} filas), historia = entrenamiento real + predicción pública",
        flush=True,
    )
    stacked = [np.zeros(len(test)) for _ in range(4)]
    public_pos = np.flatnonzero(public.to_numpy())
    private_pos = np.flatnonzero(private.to_numpy())
    for i in range(4):
        stacked[i][public_pos] = public_pred[i]
        stacked[i][private_pos] = private_pred[i]
    return tuple(stacked), state["features"], state["model"], list(state["model"].columns)
