import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tecton.demo import generate
from tecton.features import Features
from tecton.metrics import score
from tecton.pipeline import fold_indices, promotion_reasons, verify_run_artifacts
from tecton.schema import KEYS, OUTPUT, load_inputs, sha256, validate_predictions


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.temp.name)
        generate(cls.data, municipalities=12)
        cls.frames, _, _, _ = load_inputs(cls.data, strict=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_no_current_or_future_target_in_history(self):
        train = self.frames[0].copy()
        train = train.sample(frac=1, random_state=5).reset_index(drop=True)
        altered = train.copy()
        changed_date = pd.Timestamp("2020-01-01")
        altered.loc[altered.fecha >= changed_date, "tiene_evento"] = 1
        altered.loc[altered.fecha >= changed_date, "personas_desplazadas"] = 9999
        x = Features().fit_transform(train)
        x_altered = Features().fit_transform(altered)
        # Incluye TODO el mes cambiado; evita fuga de otros municipios del mismo mes.
        pd.testing.assert_frame_equal(x.loc[train.fecha <= changed_date], x_altered.loc[train.fecha <= changed_date])

    def test_inference_ignores_validation_targets(self):
        train = self.frames[0].copy()
        prefix = train[train.fecha <= "2021-09-01"]
        validation = train[train.fecha > "2021-09-01"].copy()
        model = Features()
        model.fit_transform(prefix)
        before = model.transform(validation)
        validation["tiene_evento"] = 1
        validation["personas_desplazadas"] = 9999
        pd.testing.assert_frame_equal(before, model.transform(validation))

    def test_inference_history_is_frozen(self):
        model = Features()
        model.fit_transform(self.frames[0])
        test = pd.concat(self.frames[1:], ignore_index=True)
        x = model.transform(test)
        for _, group in x.groupby("DIVIPOLA"):
            self.assertEqual(group["history_mean_log_people"].nunique(), 1)
        with self.assertRaises(ValueError):
            model.transform(self.frames[0].iloc[:1])

    def test_folds_keep_whole_months(self):
        train = self.frames[0]
        folds = [{"train_end": "2021-09-01", "valid_start": "2021-10-01", "valid_end": "2022-03-01"}]
        tr, va = fold_indices(train, folds)[0]
        self.assertLess(train.loc[tr, "fecha"].max(), train.loc[va, "fecha"].min())
        self.assertTrue(train.loc[va].groupby("fecha").size().eq(12).all())
        with self.assertRaises(ValueError):
            fold_indices(train, folds * 2)

    def test_metric_definitions(self):
        metrics = score([0, 1], [0, np.expm1(3)], (np.array([0.1, 0.9]), np.array([0, 3]), np.array([0, 0]), np.array([1, 1])))
        self.assertEqual(metrics["auc"], 1)
        self.assertAlmostEqual(metrics["rmse_log"], 0)
        self.assertAlmostEqual(metrics["winkler_log"], 11)
        self.assertEqual(metrics["coverage_80"], 0.5)
        # Nulo: RMSE sqrt(4.5), Winkler 15 -> 100 * (0.5 + 0.2 + 0.05 * (1 - 11 / 15)).
        self.assertAlmostEqual(metrics["puntos_75"], 100 * (0.5 + 0.2 + 0.05 * (4 / 15)))

    def test_prediction_keys_bounds_and_finite_values(self):
        keys = self.frames[1][KEYS].head(3).copy()
        pred = keys.copy()
        for name in OUTPUT[2:]:
            pred[name] = 0.1
        validate_predictions(pred, keys)
        for bad in [pred.iloc[:-1], pred.iloc[::-1]]:
            with self.assertRaises(ValueError):
                validate_predictions(bad, keys)
        bad = pred.copy()
        bad.loc[bad.index[0], "personas_desplazadas_q10"] = 10
        with self.assertRaises(ValueError):
            validate_predictions(bad, keys)
        bad = pred.copy()
        bad.loc[bad.index[0], "prob_evento"] = np.inf
        with self.assertRaises(ValueError):
            validate_predictions(bad, keys)
        self.assertTrue(pred.DIVIPOLA.str.startswith("0").any())

    def test_relaxed_schema_requires_synthetic_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_inputs(Path(folder), strict=False)

    def test_synthetic_test_has_no_targets(self):
        self.assertNotIn("tiene_evento", self.frames[1])
        self.assertNotIn("personas_desplazadas", self.frames[2])

    def test_synthetic_initializes_empty_scaffold_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            (destination / ".gitkeep").touch()
            generate(destination, municipalities=3)
            self.assertTrue((destination / "SYNTHETIC.json").exists())

    def test_promotion_rejects_regression_and_data_change(self):
        manifest = {"status": "complete", "qa_passed": True, "data_hashes": {"a": "1"}, "protocol_hash": "same"}
        old = {"summary": {"auc_mean": 0.8, "rmse_log_mean": 1, "winkler_log_mean": 2}, "folds": [{"auc": 0.8}] * 5}
        better = {"summary": {"auc_mean": 0.81, "rmse_log_mean": 1, "winkler_log_mean": 2}, "folds": [{"auc": 0.81}] * 5}
        self.assertEqual(promotion_reasons(better, manifest, old, manifest), [])
        self.assertTrue(promotion_reasons(old, manifest, old, manifest))
        changed = {**manifest, "data_hashes": {"a": "different"}}
        self.assertTrue(any("datos" in r for r in promotion_reasons(better, changed, old, manifest)))

    def test_file_hook_protects_raw_and_allows_source(self):
        root = Path(__file__).resolve().parents[1]
        hook = root / ".claude/hooks/protect_files.py"
        for file_path, should_deny in [("data/raw/entrenamiento.csv", True), ("runs/test/metrics.json", True), ("src/tecton/models.py", False)]:
            result = subprocess.run([sys.executable, str(hook)], input=json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(root / file_path)}}), text=True, capture_output=True, env={**os.environ, "CLAUDE_PROJECT_DIR": str(root)}, check=True)
            self.assertEqual(bool(result.stdout), should_deny)
            if should_deny:
                self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_changed_artifact_cannot_be_delivered(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            names = ["predicciones.csv", "metrics.json", "source.zip"]
            for name in names:
                (out / name).write_bytes(b"original")
            manifest = {"artifact_hashes": {name: sha256(out / name) for name in names}}
            verify_run_artifacts(out, manifest)
            (out / "predicciones.csv").write_bytes(b"edited")
            with self.assertRaises(ValueError):
                verify_run_artifacts(out, manifest)


if __name__ == "__main__":
    unittest.main()
