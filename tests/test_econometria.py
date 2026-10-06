import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tecton.demo import generate
from tecton.econometria import fit_predict_econ, prior_means
from tecton.garch import PanelGarch
from tecton.schema import load_inputs


class EconometriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.temp.name)
        generate(cls.data, municipalities=12)
        cls.frames, _, _, _ = load_inputs(cls.data, strict=False)
        cls.config = {"features": {"use_history": True, "smoothing": 20.0}, "model": {"family": "econ"}}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_prior_ignores_current_and_future_months(self):
        frame = self.frames[0]
        early = frame[frame.fecha <= "2020-09-01"]
        monthly = frame.loc[frame.fecha <= "2021-06-01", ["DIVIPOLA", "fecha", "n_inundacion"]].copy()
        before = prior_means(early, monthly[monthly.fecha <= "2020-09-01"], ["n_inundacion"])
        leaked = monthly.copy()
        leaked.loc[leaked.fecha > "2020-09-01", "n_inundacion"] = 10_000
        after = prior_means(early, leaked[leaked.fecha <= "2020-09-01"], ["n_inundacion"])
        pd.testing.assert_frame_equal(before, after)
        first = frame.fecha == frame.fecha.min()
        self.assertTrue((prior_means(frame.loc[first], frame.loc[first], ["n_inundacion"]).to_numpy() == 0).all())

    def test_hurdle_respects_fold_and_private_stage(self):
        train = self.frames[0]
        prefix = train[train.fecha <= "2021-09-01"].reset_index(drop=True)
        validation = train[train.fecha > "2021-09-01"].reset_index(drop=True)
        pred, _, _, columns = fit_predict_econ(prefix, validation, self.config, root=self.data)
        self.assertEqual(len(pred[0]), len(validation))
        self.assertTrue(np.isfinite(np.column_stack(pred)).all())
        self.assertTrue(((pred[0] > 0) & (pred[0] < 1)).all())
        self.assertIn("oficial_lag_n_inundacion", columns)
        full_test = pd.concat(self.frames[1:], ignore_index=True)
        full, _, _, _ = fit_predict_econ(train, full_test, self.config, root=self.data)
        self.assertEqual(len(full[0]), len(full_test))
        self.assertTrue((full[2] <= full[3]).all())

    def test_garch_uses_only_months_before_row_and_cutoff(self):
        train = self.frames[0]
        cutoff = pd.Timestamp("2021-09-01")
        prefix = train[train.fecha <= cutoff]
        rows = train[train.fecha.between("2020-01-01", "2022-09-01")]
        base = PanelGarch().fit(prefix, cutoff)
        leaked = train.copy()
        leaked.loc[leaked.fecha > cutoff, "personas_desplazadas"] = 10_000
        same = PanelGarch().fit(leaked, cutoff)
        pd.testing.assert_frame_equal(base.transform(rows), same.transform(rows))
        inside = rows[rows.fecha <= cutoff]
        changed = prefix.copy()
        changed.loc[changed.fecha == cutoff, "personas_desplazadas"] = 10_000
        bumped = PanelGarch().fit(changed, cutoff, params=(base.alpha, base.beta))
        same_month = inside[inside.fecha == cutoff]
        pd.testing.assert_frame_equal(base.transform(same_month), bumped.transform(same_month))
        self.assertLess(base.alpha + base.beta, 1.0)
        far = rows[rows.fecha > cutoff]
        forecast = base.transform(far)
        self.assertTrue((forecast["garch_sigma"] > 0).all())
        self.assertTrue(np.isfinite(base.transform(inside).to_numpy()).all())

    def test_features_econ_block_is_causal_and_frozen(self):
        from tecton.features import Features

        train = self.frames[0]
        cutoff = pd.Timestamp("2021-09-01")
        prefix = train[train.fecha <= cutoff].reset_index(drop=True)
        future = train[train.fecha > cutoff].reset_index(drop=True)
        econ = {"garch": True, "official_lags": True}
        model = Features(econ=econ)
        x_train = model.fit_transform(prefix)
        x_future = model.transform(future.drop(columns=["tiene_evento", "personas_desplazadas"]))
        for column in ["garch_sigma", "oficial_lag_n_inundacion"]:
            self.assertIn(column, x_train.columns)
            self.assertIn(column, x_future.columns)
        self.assertTrue(np.isfinite(x_future[["garch_sigma", "garch_ratio"]].to_numpy()).all())
        changed = prefix.copy()
        last = changed.fecha == cutoff
        changed.loc[last, "personas_desplazadas"] = 10_000
        changed.loc[last, "n_inundacion"] = 10_000
        other = Features(econ=econ)
        x_changed = other.fit_transform(changed)
        before = ~prefix.fecha.eq(cutoff)
        pd.testing.assert_series_equal(x_train.loc[last, "oficial_lag_n_inundacion"],
                                       x_changed.loc[last, "oficial_lag_n_inundacion"])
        self.assertTrue(before.any())

    def test_garch_flag_adds_variance_columns(self):
        train = self.frames[0]
        prefix = train[train.fecha <= "2021-09-01"].reset_index(drop=True)
        validation = train[train.fecha > "2021-09-01"].reset_index(drop=True)
        config = {**self.config, "model": {"family": "econ", "garch": True}}
        pred, _, _, columns = fit_predict_econ(prefix, validation, config, root=self.data)
        self.assertIn("garch_sigma", columns)
        self.assertTrue(np.isfinite(np.column_stack(pred)).all())
