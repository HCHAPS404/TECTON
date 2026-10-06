"""Motores CPU y tres objetivos. Iteraciones fijas, sin early stopping aleatorio.

Familias: hist (scikit-learn), catboost, lgbm (LightGBM) y xgb (XGBoost). LightGBM y XGBoost
vienen preinstalados en Colab; en la PC usar `uv run --with lightgbm --with xgboost`.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from tecton.features import CAT


class Models:
    def __init__(self, config):
        self.config = config

    def _build(self, x):
        params = self.config["model"]
        family = params["family"]
        iterations = params["iterations"]
        seed, threads = self.config["seed"], self.config["threads"]
        if family == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor

            common = dict(
                iterations=iterations, depth=params["depth"], learning_rate=params["learning_rate"],
                random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False, cat_features=CAT,
            )
            # Con muchos ceros exactos, la estimación Exact de hojas sesga q10 hacia arriba (cobertura ~5% en ensayo).
            return (CatBoostClassifier(loss_function="Logloss", **common), CatBoostRegressor(loss_function="RMSE", **common),
                    CatBoostRegressor(loss_function="Quantile:alpha=0.1", leaf_estimation_method="Gradient", **common),
                    CatBoostRegressor(loss_function="Quantile:alpha=0.9", leaf_estimation_method="Gradient", **common))
        # Excluir ID municipal ordinal salvo que el motor lo trate como categoría nativa.
        use_divipola = family in ["lgbm", "xgb"] and params.get("divipola_categorical", False)
        self.cat_cols = [c for c in CAT if c != "DIVIPOLA" or use_divipola]
        self.num_cols = [c for c in x.columns if c not in CAT]
        if family == "hist":
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(x[self.cat_cols])
            categorical = [False] * len(self.num_cols) + [True] * len(self.cat_cols)
            common = dict(
                max_iter=iterations, max_depth=params["depth"], learning_rate=params["learning_rate"],
                min_samples_leaf=params.get("min_samples_leaf", 20), l2_regularization=params.get("l2", 2),
                max_leaf_nodes=params.get("max_leaf_nodes", 31), max_features=params.get("max_features", 1.0),
                random_state=seed, early_stopping=False, categorical_features=categorical,
            )
            return (HistGradientBoostingClassifier(**common), HistGradientBoostingRegressor(loss="squared_error", **common),
                    HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **common),
                    HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **common))
        self.categories = {c: sorted(x[c].astype(str).unique()) for c in self.cat_cols}
        if family == "lgbm":
            import lightgbm as lgb

            common = dict(
                n_estimators=iterations, learning_rate=params["learning_rate"], num_leaves=params.get("max_leaf_nodes", 31),
                max_depth=params["depth"], min_child_samples=params.get("min_samples_leaf", 20),
                reg_lambda=params.get("l2", 2.0), colsample_bytree=params.get("max_features", 1.0),
                subsample=params.get("subsample", 1.0), subsample_freq=1 if params.get("subsample", 1.0) < 1 else 0,
                cat_smooth=params.get("cat_smooth", 10.0), random_state=seed, n_jobs=threads, verbose=-1,
            )
            return (lgb.LGBMClassifier(objective="binary", **common), lgb.LGBMRegressor(objective="regression", **common),
                    lgb.LGBMRegressor(objective="quantile", alpha=0.1, **common),
                    lgb.LGBMRegressor(objective="quantile", alpha=0.9, **common))
        if family == "xgb":
            import xgboost as xgb

            common = dict(
                n_estimators=iterations, learning_rate=params["learning_rate"], max_depth=params["depth"],
                min_child_weight=params.get("min_child_weight", 1.0), reg_lambda=params.get("l2", 2.0),
                colsample_bytree=params.get("max_features", 1.0), subsample=params.get("subsample", 1.0),
                tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, random_state=seed, n_jobs=threads,
            )
            return (xgb.XGBClassifier(objective="binary:logistic", **common), xgb.XGBRegressor(objective="reg:squarederror", **common),
                    xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.1, **common),
                    xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9, **common))
        raise ValueError(f"Motor no implementado: {family}.")

    def _encode(self, x):
        family = self.config["model"]["family"]
        if family == "catboost":
            return x
        if family == "hist":
            return np.column_stack([x[self.num_cols].to_numpy(float), self.encoder.transform(x[self.cat_cols])])
        frame = x[self.num_cols].astype(float).copy()
        for col in self.cat_cols:
            frame[col] = pd.Categorical(x[col].astype(str), categories=self.categories[col])
        return frame

    def fit_classifier_only(self, x, event):
        self.classifier = self._build(x)[0]
        self.classifier.fit(self._encode(x), event)
        return self

    def predict_proba_only(self, x):
        return self.classifier.predict_proba(self._encode(x))[:, 1]

    def fit(self, x, event, log_people):
        if len(np.unique(event)) != 2:
            raise ValueError("Se requieren ambas clases en entrenamiento.")
        self.classifier, self.point, self.low, self.high = self._build(x)
        encoded = self._encode(x)
        self.classifier.fit(encoded, event)
        self.point.fit(encoded, log_people)
        self.low.fit(encoded, log_people)
        self.high.fit(encoded, log_people)
        return self

    def predict(self, x):
        encoded = self._encode(x)
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
