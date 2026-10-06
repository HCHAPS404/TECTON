"""Historial causal en entrenamiento; historial congelado en cada validación/test."""

import numpy as np
import pandas as pd

from tecton.schema import NUMERIC

CAT = ["DIVIPOLA", "departamento", "mes_cat", "fase_enso"]
BASE_STATS = ["n", "e", "z", "p"]
# Columnas de historial que solo existen en entrenamiento; se usan como tasas previas al corte.
EVENT_TYPES = ["n_movimiento_masa", "n_inundacion", "n_vendaval", "n_creciente_subita", "n_avenida_torrencial", "n_otros"]


class Features:
    def __init__(self, use_history=True, smoothing=20.0, event_types=False):
        self.use_history = use_history
        self.smoothing = float(smoothing)
        self.event_types = bool(event_types)
        self.stats = BASE_STATS + ([f"t_{c}" for c in EVENT_TYPES] + ["t_viviendas"] if self.event_types else [])
        if self.smoothing <= 0:
            raise ValueError("smoothing debe ser positivo.")

    def _work(self, frame):
        work = frame.reset_index(drop=True).copy()
        work["departamento"] = work["DIVIPOLA"].str[:2]
        work["e"] = work["tiene_evento"].astype(float)
        work["z"] = np.log1p(work["personas_desplazadas"].astype(float))
        work["p"] = (work["personas_desplazadas"] > 0).astype(float)
        work["n"] = 1.0
        if self.event_types:
            for col in EVENT_TYPES:
                work[f"t_{col}"] = (work[col] > 0).astype(float)
            work["t_viviendas"] = ((work["viv_destruidas"] + work["viv_averiadas"]) > 0).astype(float)
        return work

    def _base(self, frame):
        x = frame[NUMERIC].astype(float).copy().reset_index(drop=True)
        # Solo medianas del entrenamiento correspondiente; CatBoost y baseline ven el mismo X.
        x = x.fillna(self.medians).fillna(0)
        month = frame["mes"].to_numpy(dtype=int)
        x["mes_sin"] = np.sin(2 * np.pi * month / 12)
        x["mes_cos"] = np.cos(2 * np.pi * month / 12)
        for col in ["densidad_poblacional", "ingresos_tributarios", "inversion_gestion_riesgo"]:
            x[f"log_{col}"] = np.log1p(x[col].clip(lower=0))
        x["oni_pendiente"] = x["ONI"] * x["pendiente_media"]
        x["DIVIPOLA"] = frame["DIVIPOLA"].astype(str).to_numpy()
        x["departamento"] = x["DIVIPOLA"].str[:2]
        x["mes_cat"] = month.astype(str)
        x["fase_enso"] = frame["fase_enso"].fillna("Desconocida").astype(str).to_numpy()
        return x

    def _prior_rows(self, work, keys):
        """Sumas estrictamente anteriores a cada fila (grupo tiene una fila por fecha)."""
        grouped = work.groupby(keys, sort=False, dropna=False)
        stats = grouped[self.stats].cumsum() - work[self.stats]
        return stats.reset_index(drop=True)

    def _prior_months(self, work, keys):
        """Excluir TODO el mes actual para global/departamento, incluyendo otros municipios."""
        grouping = keys + ["fecha"]
        cols = self.stats
        monthly = work.groupby(grouping, as_index=False)[cols].sum()
        monthly = monthly.sort_values(grouping).reset_index(drop=True)
        if keys:
            previous = monthly.groupby(keys, sort=False)[cols].cumsum()
        else:
            previous = monthly[cols].cumsum()
        monthly[cols] = previous - monthly[cols]
        return work[grouping].merge(monthly, on=grouping, how="left", validate="many_to_one")[cols]

    def _history(self, global_stats, dept, muni, seasonal):
        alpha = self.smoothing
        # Priors fijos y explícitos solo para el arranque sin historia.
        g_rate = (global_stats.e + alpha * 0.05) / (global_stats.n + alpha)
        g_mean = global_stats.z / (global_stats.n + alpha)
        g_pos = (global_stats.p + alpha * 0.03) / (global_stats.n + alpha)
        d_rate = (dept.e + alpha * g_rate) / (dept.n + alpha)
        m_rate = (muni.e + alpha * d_rate) / (muni.n + alpha)
        result = pd.DataFrame({
            "history_months": muni.n,
            "history_event_rate": m_rate,
            "history_department_rate": d_rate,
            "history_seasonal_rate": (seasonal.e + alpha * m_rate) / (seasonal.n + alpha),
            "history_mean_log_people": (muni.z + alpha * g_mean) / (muni.n + alpha),
            "history_positive_people_rate": (muni.p + alpha * g_pos) / (muni.n + alpha),
        })
        if self.event_types:
            # Tasa municipal previa de cada tipo de evento, suavizada hacia la tasa global previa.
            for col in [c for c in self.stats if c.startswith("t_")]:
                g_type = (global_stats[col] + alpha * 0.01) / (global_stats.n + alpha)
                result[f"history_rate_{col[2:]}"] = (muni[col] + alpha * g_type) / (muni.n + alpha)
        # No usar el número de meses como tendencia que no existe en test futuro.
        return result.drop(columns="history_months")

    def fit_transform(self, frame):
        self.medians = frame[NUMERIC].median()
        work = self._work(frame)
        work["_row"] = np.arange(len(work))
        work = work.sort_values(["fecha", "DIVIPOLA"]).reset_index(drop=True)
        x = self._base(work)
        if self.use_history:
            history = self._history(
                self._prior_months(work, []), self._prior_months(work, ["departamento"]),
                self._prior_rows(work, ["DIVIPOLA"]), self._prior_rows(work, ["DIVIPOLA", "mes"]),
            )
            x = pd.concat([x, history], axis=1)
        # Guardar estadísticas hasta el corte para cualquier horizonte de inferencia.
        self.global_stats = work[self.stats].sum()
        self.tables = {
            "dept": work.groupby("departamento")[self.stats].sum().reset_index(),
            "muni": work.groupby("DIVIPOLA")[self.stats].sum().reset_index(),
            "seasonal": work.groupby(["DIVIPOLA", "mes"])[self.stats].sum().reset_index(),
        }
        self.cutoff = work["fecha"].max()
        x["_row"] = work["_row"].to_numpy()
        return x.sort_values("_row").drop(columns="_row").reset_index(drop=True)

    def transform(self, frame):
        if (frame["fecha"] <= self.cutoff).any():
            raise ValueError("Inferencia requiere fechas estrictamente posteriores al corte.")
        x = self._base(frame)
        if self.use_history:
            keys = frame[["DIVIPOLA", "mes"]].reset_index(drop=True).copy()
            keys["departamento"] = keys["DIVIPOLA"].str[:2]
            tables = []
            for label, cols in [("dept", ["departamento"]), ("muni", ["DIVIPOLA"]), ("seasonal", ["DIVIPOLA", "mes"])]:
                stats = keys[cols].merge(self.tables[label], on=cols, how="left", validate="many_to_one")
                tables.append(stats[self.stats].fillna(0))
            global_stats = pd.DataFrame([self.global_stats.to_dict()] * len(frame))
            x = pd.concat([x, self._history(global_stats, *tables)], axis=1)
        return x
