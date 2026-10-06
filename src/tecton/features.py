"""Historial causal en entrenamiento; historial congelado en cada validación/test."""

import numpy as np
import pandas as pd

from tecton.schema import NUMERIC

CAT = ["DIVIPOLA", "departamento", "mes_cat", "fase_enso"]
BASE_STATS = ["n", "e", "z", "p"]
# Columnas de historial que solo existen en entrenamiento; se usan como tasas previas al corte.
EVENT_TYPES = ["n_movimiento_masa", "n_inundacion", "n_vendaval", "n_creciente_subita", "n_avenida_torrencial", "n_otros"]


class Features:
    # Contextos adicionales de historial: clave de agrupación, tipo de exclusión y tasa a la que se suaviza.
    CONTEXTS = {
        "enso": (["DIVIPOLA", "fase_key"], "rows", "muni"),
        "dept_season": (["departamento", "mes"], "months", "dept"),
        "season": (["mes"], "months", "global"),
        "dept_enso": (["departamento", "fase_key"], "months", "dept"),
        # Vecindad espacial (k cabeceras más cercanas); se calcula aparte con coordenadas DANE.
        "spatial": ([], "spatial", "dept"),
        "spatial_season": ([], "spatial", "dept"),
    }

    def __init__(self, use_history=True, smoothing=20.0, event_types=False, extra_history=(), drop=(),
                 external_tables=None, spatial_k=0, external_derived=False, oni_lags=(), oni_series=None,
                 exposure_proxy=False, econ=None):
        # Bloque econométrico: {"garch": bool, "official_lags": bool, "dane": DataFrame, "ungrd": DataFrame}.
        self.econ = dict(econ or {})
        self.exposure_proxy = bool(exposure_proxy)
        # Serie nacional ONI de los archivos oficiales (entrenamiento + prueba): covariable entregada, no target.
        self.oni_lags = list(oni_lags)
        self.oni_series = oni_series or {}
        self.external_derived = bool(external_derived)
        self.spatial_k = int(spatial_k)
        # Tablas externas ya cargadas por el pipeline: {nombre: DataFrame con DIVIPOLA [y mes]}.
        self.external_tables = external_tables or {}
        self.drop = list(drop)
        self.extra_history = list(extra_history)
        unknown = set(self.extra_history) - set(self.CONTEXTS)
        if unknown:
            raise ValueError(f"Historial adicional desconocido: {sorted(unknown)}")
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
        work["fase_key"] = work["fase_enso"].fillna("Desconocida").astype(str)
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
        if self.exposure_proxy:
            # Tamaño de exposición: área aproximada por separación entre cabeceras y población = densidad x área.
            area = self._area_proxy()
            km2 = x["DIVIPOLA"].map(area).to_numpy(float)
            x["area_proxy_km2"] = km2
            x["log_poblacion_proxy"] = np.log1p(x["densidad_poblacional"].clip(lower=0).to_numpy() * km2)
        if self.oni_lags:
            series = pd.Series(self.oni_series, dtype=float)
            series.index = pd.to_datetime(series.index)
            dates = pd.to_datetime(frame["fecha"]).reset_index(drop=True)
            for lag in self.oni_lags:
                # lag > 0: meses anteriores; lag < 0: meses siguientes, ambos presentes en los archivos entregados.
                shifted = dates - pd.DateOffset(months=lag)
                x[f"oni_lag{lag}" if lag > 0 else f"oni_lead{-lag}"] = shifted.map(series).to_numpy(float)
            past = [lag for lag in self.oni_lags if lag > 0]
            if past:
                x["oni_tendencia"] = x["ONI"] - x[f"oni_lag{max(past)}"]
        for name, table in self.external_tables.items():
            keys = [k for k in ["DIVIPOLA", "mes"] if k in table.columns]
            left = pd.DataFrame({"DIVIPOLA": x["DIVIPOLA"].to_numpy(), "mes": month})
            values = [c for c in table.columns if c not in keys and pd.api.types.is_numeric_dtype(table[c])]
            joined = left[keys].merge(table[keys + values], on=keys, how="left", validate="many_to_one")
            for col in values:
                x[f"ext_{name}_{col}"] = joined[col].to_numpy(float)
                if self.external_derived and "mes" in keys:
                    # Perfil estacional: total anual del municipio y fracción del año que cae en este mes.
                    annual = table.groupby("DIVIPOLA")[col].sum()
                    total = x["DIVIPOLA"].map(annual).to_numpy(float)
                    x[f"ext_{name}_{col}_anual"] = total
                    x[f"ext_{name}_{col}_fraccion"] = np.where(total > 0, joined[col].to_numpy(float) / np.maximum(total, 1e-9), 0.0)
        # Excluir covariables por nombre exacto o prefijo (p. ej. "log_ingresos").
        return x.drop(columns=[c for c in x.columns if any(c == d or c.startswith(d + "_") or c == "log_" + d for d in self.drop)])

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

    def _history(self, global_stats, dept, muni, seasonal, extra=None):
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
        for name, stats in (extra or {}).items():
            base = {"muni": m_rate, "dept": d_rate, "global": g_rate}[self.CONTEXTS[name][2]]
            result[f"history_{name}_rate"] = (stats.e + alpha * base) / (stats.n + alpha)
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
            extra = {}
            for name in self.extra_history:
                keys, mode, _ = self.CONTEXTS[name]
                extra[name] = self._prior_rows(work, keys) if mode == "rows" else self._prior_months(work, keys)
            muni_prior, seasonal_prior = self._prior_rows(work, ["DIVIPOLA"]), self._prior_rows(work, ["DIVIPOLA", "mes"])
            if self.spatial_k:
                extra.update(self._spatial_fit(work, muni_prior, seasonal_prior))
            history = self._history(
                self._prior_months(work, []), self._prior_months(work, ["departamento"]),
                muni_prior, seasonal_prior, extra,
            )
            x = pd.concat([x, history], axis=1)
        if self.econ:
            x = pd.concat([x, self._econ(work, fit=True)], axis=1)
        # Guardar estadísticas hasta el corte para cualquier horizonte de inferencia.
        self.global_stats = work[self.stats].sum()
        self.tables = {
            "dept": work.groupby("departamento")[self.stats].sum().reset_index(),
            "muni": work.groupby("DIVIPOLA")[self.stats].sum().reset_index(),
            "seasonal": work.groupby(["DIVIPOLA", "mes"])[self.stats].sum().reset_index(),
        }
        for name in self.extra_history:
            keys = self.CONTEXTS[name][0]
            self.tables[name] = work.groupby(keys)[self.stats].sum().reset_index()
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
            keys["fase_key"] = frame["fase_enso"].fillna("Desconocida").astype(str).to_numpy()

            def lookup(label, cols):
                stats = keys[cols].merge(self.tables[label], on=cols, how="left", validate="many_to_one")
                return stats[self.stats].fillna(0)

            tables = [lookup(label, cols) for label, cols in [("dept", ["departamento"]), ("muni", ["DIVIPOLA"]), ("seasonal", ["DIVIPOLA", "mes"])]]
            extra = {name: lookup(name, self.CONTEXTS[name][0]) for name in self.extra_history}
            if self.spatial_k:
                extra.update(self._spatial_transform(keys))
            global_stats = pd.DataFrame([self.global_stats.to_dict()] * len(frame))
            x = pd.concat([x, self._history(global_stats, *tables, extra)], axis=1)
        if self.econ:
            x = pd.concat([x, self._econ(frame.reset_index(drop=True), fit=False)], axis=1)
        return x

    def _econ(self, frame, fit):
        """Solo meses <= corte. En entrenamiento, cada fila ve meses estrictamente anteriores a ella."""
        from tecton.econometria import OFFICIAL_COUNTS, prior_means
        from tecton.fuentes_publicas import CLIMATE, PUBLIC_CAP
        from tecton.garch import PanelGarch

        if fit:
            self.cutoff = frame["fecha"].max()
        parts = []
        if self.econ.get("garch"):
            if fit:
                self.garch_model = PanelGarch().fit(frame, self.cutoff)
            parts.append(self.garch_model.transform(frame).reset_index(drop=True))
        if self.econ.get("official_lags"):
            if fit:
                self.official = frame[["DIVIPOLA", "fecha", *OFFICIAL_COUNTS]].copy()
            parts.append(prior_means(frame, self.official, OFFICIAL_COUNTS).add_prefix("oficial_").reset_index(drop=True))
        ungrd = self.econ.get("ungrd")
        if ungrd is not None and len(ungrd):
            public = ungrd[ungrd["fecha"] <= min(self.cutoff, PUBLIC_CAP)]
            parts.append(prior_means(frame, public, list(CLIMATE)).add_prefix("publico_").reset_index(drop=True))
        dane = self.econ.get("dane")
        if dane is not None and len(dane):
            keys = pd.DataFrame({"DIVIPOLA": frame["DIVIPOLA"].astype(str).to_numpy(),
                                 "anio": pd.to_datetime(frame["fecha"]).dt.year.clip(upper=2022).to_numpy()})
            table = dane.assign(DIVIPOLA=dane["DIVIPOLA"].astype(str).str.zfill(5), anio=dane["anio"].astype(int))
            level = keys.merge(table[["DIVIPOLA", "anio", "poblacion"]], on=["DIVIPOLA", "anio"], how="left",
                               validate="many_to_one")["poblacion"]
            level = np.log1p(level.astype(float))
            if fit:
                self.dane_fill = float(level.median()) if level.notna().any() else 0.0
            parts.append(pd.DataFrame({"dane_log_poblacion": level.fillna(self.dane_fill).to_numpy()}))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(frame)))

    def _neighbors(self, munis):
        coords = self.external_tables.get("divipola_coords")
        if coords is None:
            raise ValueError("spatial_k requiere data/external/divipola_coords.csv en external_files.")
        table = coords.set_index("DIVIPOLA").reindex(munis)
        if table[["lat", "lon"]].isna().any().any():
            raise ValueError("Faltan coordenadas para municipios del entrenamiento.")
        lat, lon = np.radians(table["lat"].to_numpy()), np.radians(table["lon"].to_numpy())
        # Distancia equirectangular: suficiente para ordenar vecinos dentro de Colombia.
        dx = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2)
        distance = np.hypot(dx, lat[:, None] - lat[None, :])
        np.fill_diagonal(distance, np.inf)
        nearest = np.argsort(distance, axis=1)[:, : self.spatial_k]
        weights = np.zeros((len(munis), len(munis)))
        np.put_along_axis(weights, nearest, 1.0, axis=1)
        return weights

    def _spatial_fit(self, work, muni_prior, seasonal_prior):
        """Sumas previas de los vecinos en cada mes; cada vecino excluye su propio mes actual."""
        self.spatial_munis = sorted(work["DIVIPOLA"].unique())
        self.spatial_weights = self._neighbors(self.spatial_munis)
        col = work["DIVIPOLA"].map({m: i for i, m in enumerate(self.spatial_munis)}).to_numpy()
        dates = np.sort(work["fecha"].unique())
        row = np.searchsorted(dates, work["fecha"].to_numpy())
        result = {}
        for name, prior in [("spatial", muni_prior), ("spatial_season", seasonal_prior)]:
            sums = {}
            for stat in ["n", "e"]:
                grid = np.zeros((len(dates), len(self.spatial_munis)))
                grid[row, col] = prior[stat].to_numpy()
                sums[stat] = (grid @ self.spatial_weights.T)[row, col]
            result[name] = pd.DataFrame(sums)
        return result

    def _spatial_transform(self, keys):
        """Vecinos con las tablas congeladas al corte."""
        index = {m: i for i, m in enumerate(self.spatial_munis)}
        col = keys["DIVIPOLA"].map(index)
        known = col.notna().to_numpy()
        col = col.fillna(0).astype(int).to_numpy()
        totals = self.tables["muni"].set_index("DIVIPOLA").reindex(self.spatial_munis).fillna(0)
        seasonal = self.tables["seasonal"].set_index(["DIVIPOLA", "mes"])
        result = {}
        muni_sums = {stat: self.spatial_weights @ totals[stat].to_numpy() for stat in ["n", "e"]}
        result["spatial"] = pd.DataFrame({stat: np.where(known, muni_sums[stat][col], 0.0) for stat in ["n", "e"]})
        month = keys["mes"].to_numpy()
        season = {}
        for stat in ["n", "e"]:
            values = np.zeros(len(keys))
            for m in np.unique(month):
                vector = seasonal[stat].reindex(pd.MultiIndex.from_product([self.spatial_munis, [m]])).fillna(0).to_numpy()
                neighbor = self.spatial_weights @ vector
                mask = month == m
                values[mask] = np.where(known[mask], neighbor[col[mask]], 0.0)
            season[stat] = values
        result["spatial_season"] = pd.DataFrame(season)
        return result

    def _area_proxy(self):
        if getattr(self, "_area_cache", None) is None:
            coords = self.external_tables.get("divipola_coords")
            if coords is None:
                raise ValueError("exposure_proxy requiere data/external/divipola_coords.csv en external_files.")
            lat, lon = np.radians(coords["lat"].to_numpy()), np.radians(coords["lon"].to_numpy())
            dx = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2)
            km = np.hypot(dx, lat[:, None] - lat[None, :]) * 6371.0
            np.fill_diagonal(km, np.inf)
            mean3 = np.sort(km, axis=1)[:, :3].mean(axis=1)
            self._area_cache = dict(zip(coords["DIVIPOLA"], mean3 ** 2))
        return self._area_cache
