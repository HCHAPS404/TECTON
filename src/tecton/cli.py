import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

from tecton.pipeline import dump_json, promote, resolve_run, run
from tecton.schema import load_inputs


def main():
    parser = argparse.ArgumentParser(description="Harness TECTON: experimentos y entregas locales")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument("--municipalities", type=int, default=20)
    audit = sub.add_parser("audit")
    audit.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    execute = sub.add_parser("run")
    execute.add_argument("--config", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("run_ids", nargs="*")
    promote_parser = sub.add_parser("promote")
    promote_parser.add_argument("run_id")
    promote_parser.add_argument("--reason", required=True)
    promote_parser.add_argument("--demo", action="store_true")
    export = sub.add_parser("bundle")
    export.add_argument("--run-id")
    export.add_argument("--demo", action="store_true")
    dashboard = sub.add_parser("dashboard", help="Exportar geovisor local desde un run validado")
    dashboard.add_argument("--run-id")
    dashboard.add_argument("--geojson", type=Path)
    dashboard.add_argument("--out", type=Path)
    dashboard.add_argument("--demo", action="store_true")
    geography = sub.add_parser("geography", help="Descargar polígonos municipales abiertos de DANE")
    geography.add_argument("--out", type=Path, default=Path("data/geography/municipios.geojson"))
    notebook = sub.add_parser("notebook")
    notebook.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    sub.add_parser("freeze")
    sub.add_parser("clock")
    record = sub.add_parser("record-submission")
    record.add_argument("run_id")
    record.add_argument("--received-at", help="Hora confirmada por el formulario con zona ISO8601; default=ahora")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "doctor":
            modules = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "catboost"]}
            print(json.dumps({"python": platform.python_version(), "packages": modules, "root": str(root), "confidential_data_uploaded": False}, indent=2))
            if not (3, 11) <= sys.version_info[:2] < (3, 14):
                raise ValueError("Python fuera del rango probado/configurado.")
            for name in ["configs/baseline.json", "configs/catboost.json", "requirements-colab.txt"]:
                if not (root / name).exists():
                    raise ValueError(f"Falta {name}")
        elif args.command == "synthetic":
            from tecton.demo import generate

            generate(root / "data/synthetic", args.municipalities)
            print("Datos inventados creados en data/synthetic; solo ensayo.")
        elif args.command == "audit":
            config = json.loads((root / args.config).read_text())
            _, summary, _, synthetic = load_inputs(root / config["data_dir"], config["strict"])
            print(json.dumps({"synthetic": synthetic, "audit": summary}, indent=2))
        elif args.command == "run":
            run(root, root / args.config)
        elif args.command == "compare":
            paths = [resolve_run(root, rid) for rid in args.run_ids] if args.run_ids else sorted((root / "runs").glob("*/metrics.json"))
            for path in paths:
                path = path.parent if path.is_file() else path
                manifest = json.loads((path / "manifest.json").read_text())
                if manifest["status"] == "complete":
                    metrics = json.loads((path / "metrics.json").read_text())
                    print(json.dumps({"run_id": path.name, "synthetic": manifest["synthetic"], "protocol_hash": manifest["protocol_hash"], **metrics["summary"]}, ensure_ascii=False))
        elif args.command == "promote":
            print(json.dumps(promote(root, args.run_id, args.reason, args.demo), ensure_ascii=False, indent=2))
        elif args.command == "bundle":
            from tecton.artifacts import bundle

            print(bundle(root, args.run_id, args.demo))
        elif args.command == "dashboard":
            from tecton.dashboard import export_dashboard

            print(export_dashboard(root, args.run_id, args.geojson, args.out, args.demo))
        elif args.command == "geography":
            from tecton.dashboard import download_geography

            print(download_geography(root / args.out))
        elif args.command == "notebook":
            from tecton.artifacts import portable_notebook, snapshot

            target = root / "notebooks/tecton_colab.ipynb"
            source = root / "notebooks/source-template.zip"
            source.parent.mkdir(parents=True, exist_ok=True)
            snapshot(root, source)
            portable_notebook(source, json.loads((root / args.config).read_text()), target)
            source.unlink()
            print(target)
        elif args.command == "freeze":
            pointer = root / "state/champion.json"
            if not pointer.exists():
                raise ValueError("Promover un champion real antes de congelar.")
            dump_json(root / "state/FROZEN.json", {"run_id": json.loads(pointer.read_text())["run_id"], "frozen_at": datetime.now(ZoneInfo("America/Bogota")).isoformat()})
            print("Champion congelado. bundle continúa funcionando; nuevos runs son candidatos independientes.")
        elif args.command == "record-submission":
            resolve_run(root, args.run_id)
            received = datetime.fromisoformat(args.received_at) if args.received_at else datetime.now(ZoneInfo("America/Bogota"))
            if received.tzinfo is None:
                raise ValueError("Hora debe incluir zona, ej. 2026-10-06T10:30:00-05:00.")
            path = root / "state/submissions.jsonl"
            path.parent.mkdir(exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"run_id": args.run_id, "received_at": received.isoformat()}) + "\n")
            print("Registro local agregado. Esto NO envía el formulario oficial.")
        elif args.command == "clock":
            now = datetime.now(ZoneInfo("America/Bogota"))
            deadline = datetime.fromisoformat("2026-10-06T14:00:00-05:00")
            path = root / "state/submissions.jsonl"
            submitted = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            earliest = max(datetime.fromisoformat(row["received_at"]) for row in submitted) + timedelta(minutes=30) if submitted else now
            print(json.dumps({"now_bogota": now.isoformat(), "deadline": deadline.isoformat(), "earliest_next_submission": earliest.isoformat(), "minutes_to_deadline": round((deadline - now).total_seconds() / 60, 1)}, indent=2))
    except (ValueError, FileNotFoundError, HTTPError, URLError) as error:
        parser.exit(2, f"Error: {error}\n")
