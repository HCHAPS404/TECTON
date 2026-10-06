"""Dos motores CPU y tres objetivos. Iteraciones fijas, sin early stopping aleatorio."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from tecton.features import CAT


class Models:
    def __init__(self, config):
        self.config = config

    def fit(self, x, event, log_people):
        params = self.config["model"]
        family = params["family"]
        iterations = params["iterations"]
        if family == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor

            common = dict(
                iterations=iterations, depth=params["depth"], learning_rate=params["learning_rate"],
                random_seed=self.config["seed"], thread_count=self.config["threads"],
                verbose=False, allow_writing_files=False, cat_features=CAT,
            )
            self.classifier = CatBoostClassifier(loss_function="Logloss", **common)
            self.point = CatBoostRegressor(loss_function="RMSE", **common)
            # Con muchos ceros exactos, la estimación Exact de hojas sesga q10 hacia arriba (cobertura ~5% en ensayo).
            self.low = CatBoostRegressor(loss_function="Quantile:alpha=0.1", leaf_estimation_method="Gradient", **common)
            self.high = CatBoostRegressor(loss_function="Quantile:alpha=0.9", leaf_estimation_method="Gradient", **common)
            encoded = x
        elif family == "hist":
            # Excluir ID municipal ordinal: su orden arbitrario no representa proximidad.
            self.num_cols = [c for c in x.columns if c not in CAT]
            self.cat_cols = [c for c in CAT if c != "DIVIPOLA"]
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.encoder.fit(x[self.cat_cols])
            encoded = self._encode(x)
            categorical = [False] * len(self.num_cols) + [True] * len(self.cat_cols)
            common = dict(
                max_iter=iterations, max_depth=params["depth"], learning_rate=params["learning_rate"],
                min_samples_leaf=params.get("min_samples_leaf", 20), l2_regularization=params.get("l2", 2),
                max_leaf_nodes=params.get("max_leaf_nodes", 31), max_features=params.get("max_features", 1.0),
                random_state=self.config["seed"],
                early_stopping=False, categorical_features=categorical,
            )
            self.classifier = HistGradientBoostingClassifier(**common)
            self.point = HistGradientBoostingRegressor(loss="squared_error", **common)
            self.low = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **common)
            self.high = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **common)
        else:
            raise ValueError(f"Motor no implementado: {family}.")
        if len(np.unique(event)) != 2:
            raise ValueError("Se requieren ambas clases en entrenamiento.")
        self.classifier.fit(encoded, event)
        self.point.fit(encoded, log_people)
        self.low.fit(encoded, log_people)
        self.high.fit(encoded, log_people)
        return self

    def _encode(self, x):
        return np.column_stack([x[self.num_cols].to_numpy(float), self.encoder.transform(x[self.cat_cols])])

    def predict(self, x):
        encoded = self._encode(x) if self.config["model"]["family"] == "hist" else x
        prob = self.classifier.predict_proba(encoded)[:, 1]
        point = np.maximum(0, self.point.predict(encoded))
        raw_low = np.maximum(0, self.low.predict(encoded))
        raw_high = np.maximum(0, self.high.predict(encoded))
        low = np.minimum(raw_low, raw_high)
        high = np.maximum(raw_low, raw_high)
        result = np.column_stack([prob, point, low, high])
        if not np.isfinite(result).all():
            raise ValueError("Modelo produjo valores no finitos.")
        if (result[:, 1:] > np.log(np.finfo(float).max) - 1).any():
            raise ValueError("Predicción log demasiado grande; revisar experimento.")
        return prob, point, low, high
