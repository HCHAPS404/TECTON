import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tecton.artifacts import import_run, portable_notebook, review_pack, run_pack, snapshot
from tecton.demo import generate
from tecton.pipeline import compare_runs, run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "name": "colab-test", "data_dir": "data/synthetic", "strict": False, "seed": 42, "threads": 1,
    "model": {"family": "hist", "iterations": 5, "learning_rate": 0.1, "depth": 3},
    "features": {"use_history": True, "smoothing": 20.0}, "stress_test": False,
    "folds": [{"train_end": "2021-09-01", "valid_start": "2021-10-01", "valid_end": "2022-09-01"}],
}


class ColabFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "colab"
        generate(cls.root / "data/synthetic", municipalities=6)
        (cls.root / "configs").mkdir(parents=True)
        (cls.root / "configs/test.json").write_text(json.dumps(CONFIG))
        cls.run_id = run(cls.root, cls.root / "configs/test.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_notebook_cells_compile_and_expose_selector(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.zip", Path(folder) / "nb.ipynb"
            snapshot(ROOT, source)
            with zipfile.ZipFile(source) as archive:
                self.assertIn("docs/fuentes_datos.csv", archive.namelist())
                self.assertFalse(any(n.startswith("data/") and not n.startswith("data/external/") for n in archive.namelist()))
            portable_notebook(source, CONFIG, target)
            notebook = json.loads(target.read_text())
            code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            for i, cell in enumerate(code):
                compile(cell, f"cell-{i}", "exec")
            joined = "\n".join(code)
            for marker in ["CONFIG_NAME = None", "URL_DATOS = ''", "review_pack", "run_pack", "compare_runs"]:
                self.assertIn(marker, joined)
            self.assertNotIn("USE_SYNTHETIC_DATA", joined)
            portable_notebook(source, CONFIG, target, "https://drive.google.com/file/d/abc/view")
            self.assertIn("URL_DATOS = 'https://drive.google.com/file/d/abc/view'", target.read_text())

    def test_review_pack_has_only_aggregates(self):
        with zipfile.ZipFile(review_pack(self.root)) as archive:
            names = archive.namelist()
            compared = json.loads(archive.read("resumen/compare.json"))
        self.assertIn(f"resumen/runs/{self.run_id}/metrics.json", names)
        for forbidden in ["oof.csv", "predicciones.csv", "model.joblib", "source.zip"]:
            self.assertFalse(any(Path(n).name == forbidden for n in names), forbidden)
        self.assertFalse(any(n.startswith("resumen/data") for n in names))
        self.assertEqual(compared[0]["run_id"], self.run_id)
        self.assertIn("seconds", json.loads((self.root / "runs" / self.run_id / "metrics.json").read_text())["folds"][0])

    def test_run_pack_excludes_oof_and_imports_with_hash_check(self):
        archive = run_pack(self.root, self.run_id)
        with zipfile.ZipFile(archive) as zipped:
            self.assertFalse(any(Path(n).name == "oof.csv" for n in zipped.namelist()))
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            self.assertEqual(import_run(local, archive), self.run_id)
            self.assertEqual(compare_runs(local)[0]["run_id"], self.run_id)
            with self.assertRaises(ValueError):
                import_run(local, archive)
        tampered = archive.with_name("tampered.zip")
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, "w") as target:
            for item in source.infolist():
                data = source.read(item)
                target.writestr(item, b"DIVIPOLA,editado\n" if item.filename.endswith("predicciones.csv") else data)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                import_run(Path(folder), tampered)
            self.assertFalse(any((Path(folder) / "runs").iterdir()))


if __name__ == "__main__":
    unittest.main()
