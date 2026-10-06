import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from tecton import calibration
from tecton.features import Features
from tecton.metrics import score
from tecton.models import Models
from tecton.schema import KEYS, OUTPUT, load_inputs, sha256, validate_predictions

METRIC_PROTOCOL = {"version": 3, "risk": "AUC-ROC", "magnitude": "RMSE-log1p-all-rows", "interval": "Winkler-log1p-alpha0.2", "points": "notebook-base-PNUD-null-normalized-75"}


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fold_indices(train, folds):
    seen = set()
    result = []
    for fold in folds:
        cutoff, start, end = (pd.Timestamp(fold[c]) for c in ["train_end", "valid_start", "valid_end"])
        if not cutoff < start <= end:
            raise ValueError("Fold temporal inválido.")
        tr = train["fecha"] <= cutoff
        va = train["fecha"].between(start, end)
        if not tr.any() or not va.any():
            raise ValueError("Fold vacío; revisar los períodos.")
        months = set(train.loc[va, "fecha"])
        if seen & months:
            raise ValueError("Los folds primarios no deben solapar meses de validación.")
        seen |= months
        if set(train.loc[tr, "fecha"]) & months:
            raise ValueError("Un mes está simultáneamente en entrenamiento y validación.")
        result.append((tr, va))
    if not result:
        raise ValueError("Se requiere al menos un fold.")
    return result


def _months_after(cutoff, dates):
    return ((dates.dt.year - cutoff.year) * 12 + dates.dt.month - cutoff.month).to_numpy(float)


def _frozen_training(train, config, options):
    """Filas de entrenamiento con historial congelado en cortes simulados, como en la prueba real.

    Para cada corte c, las filas de los meses (c, c + horizonte] reciben features calculadas solo
    con datos <= c. Así el modelo aprende con el mismo historial envejecido que verá en test.
    """
    start = train["fecha"].min() + pd.DateOffset(months=options.get("min_history_months", 12))
    end = train["fecha"].max() - pd.DateOffset(months=1)
    horizon = options.get("horizon_months", 24)
    parts = []
    for cutoff in pd.date_range(start, end, freq=f"{options.get('every_months', 6)}MS"):
        rows = train[(train["fecha"] > cutoff) & (train["fecha"] <= cutoff + pd.DateOffset(months=horizon))]
        model = Features(**config["features"])
        model.fit_transform(train[train["fecha"] <= cutoff])
        x = model.transform(rows)
        if options.get("horizon_feature", True):
            x["horizonte_meses"] = _months_after(cutoff, rows["fecha"])
        parts.append((x, rows["tiene_evento"].to_numpy(int), np.log1p(rows["personas_desplazadas"].to_numpy(float))))
    if not parts:
        raise ValueError("Sin cortes congelados: entrenamiento demasiado corto para la configuración.")
    return (pd.concat([p[0] for p in parts], ignore_index=True), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def _fit_predict_raw(train, test, config):
    feature_model = Features(**config["features"])
    x_train = feature_model.fit_transform(train)
    x_test = feature_model.transform(test)
    y = train["tiene_evento"].to_numpy(int)
    z = np.log1p(train["personas_desplazadas"].to_numpy(float))
    warmup = config.get("training", {}).get("warmup_months", 0)
    frozen = config.get("training", {}).get("frozen_cutoffs")
    if frozen:
        x_fit, y, z = _frozen_training(train, config, frozen)
        if frozen.get("horizon_feature", True):
            x_test = x_test.assign(horizonte_meses=_months_after(feature_model.cutoff, test["fecha"].reset_index(drop=True)))
        x_train = x_fit
    elif warmup:
        # Los primeros meses casi no tienen historial previo; sirven para construirlo, no para ajustar.
        keep = (train["fecha"] >= train["fecha"].min() + pd.DateOffset(months=warmup)).to_numpy()
        x_fit, y, z = x_train.loc[keep].reset_index(drop=True), y[keep], z[keep]
    else:
        x_fit = x_train
    seeds = config.get("training", {}).get("seeds", [config["seed"]])
    models = [Models({**config, "seed": seed}).fit(x_fit, y, z) for seed in seeds]
    outputs = [m.predict(x_test) for m in models]
    predictions = tuple(np.mean([o[i] for o in outputs], axis=0) for i in range(4))
    model = models[0]
    model.members = models[1:]
    return predictions, feature_model, model, list(x_train.columns)


def fit_predict(train, test, config):
    predictions, feature_model, model, features = _fit_predict_raw(train, test, config)
    months = config.get("calibration", {}).get("months", 0)
    if months:
        # Ajustar con los últimos meses del MISMO entrenamiento, usando un modelo que no los vio.
        cutoff = train["fecha"].max() - pd.DateOffset(months=months)
        inner_train, inner_valid = train[train["fecha"] <= cutoff], train[train["fecha"] > cutoff]
        inner, _, _, _ = _fit_predict_raw(inner_train, inner_valid, config)
        model.calibration = calibration.fit(np.log1p(inner_valid["personas_desplazadas"].to_numpy(float)), *inner)
        predictions = calibration.apply(model.calibration, predictions)
    return predictions, feature_model, model, features


def with_external(root, config, codes):
    """Devuelve una copia de config con las tablas externas cargadas y validadas."""
    files = config["features"].get("external_files", [])
    tables = {}
    for relative in files:
        table = pd.read_csv(Path(root) / relative, dtype={"DIVIPOLA": "string"})
        keys = [k for k in ["DIVIPOLA", "mes"] if k in table]
        if not table["DIVIPOLA"].str.fullmatch(r"\d{5}").all() or table.duplicated(keys).any():
            raise ValueError(f"{relative}: DIVIPOLA inválido o llaves duplicadas.")
        if set(codes) - set(table["DIVIPOLA"]):
            raise ValueError(f"{relative}: faltan {len(set(codes) - set(table['DIVIPOLA']))} municipios del reto.")
        tables[Path(relative).stem] = table
    features = {k: v for k, v in config["features"].items() if k != "external_files"}
    return {**config, "features": {**features, "external_tables": tables}}


def run(root: Path, config_path: Path):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Caminos siempre relativos al proyecto, iguales en local y Colab.
    data_dir = root / config["data_dir"]
    frames, audit, hashes, synthetic = load_inputs(data_dir, config["strict"])
    train = frames[0]
    tests = pd.concat(frames[1:], ignore_index=True)
    split_indices = fold_indices(train, config["folds"])
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", config["name"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + slug
    out = root / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = {**METRIC_PROTOCOL, "folds": config["folds"], "stress_test": config.get("stress_test", False)}
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]}
    try:
        versions["catboost"] = importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        pass
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    manifest = {
        "run_id": run_id, "status": "running", "synthetic": synthetic,
        "config": config, "data_hashes": hashes, "protocol": protocol,
        "protocol_hash": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(out / "manifest.json", manifest)
    dump_json(out / "audit.json", audit)
    dump_json(out / "config.json", config)
    # Capturar el código ANTES de entrenar; el bundle reproduce esta versión.
    from tecton.artifacts import snapshot

    snapshot(root, out / "source.zip")
    manifest["source_sha256"] = sha256(out / "source.zip")
    dump_json(out / "manifest.json", manifest)
    # Tablas externas declaradas: se cargan una vez, se registran con hash y viajan con el modelo.
    if config["features"].get("external_files"):
        manifest["external_hashes"] = {rel: sha256(root / rel) for rel in config["features"]["external_files"]}
        dump_json(out / "manifest.json", manifest)
        config = with_external(root, config, set(train["DIVIPOLA"]))
    folds, oof = [], []
    try:
        with threadpool_limits(limits=config["threads"]):
            for i, (tr_mask, va_mask) in enumerate(split_indices, 1):
                fold_started = time.perf_counter()
                training, validation = train.loc[tr_mask].copy(), train.loc[va_mask].copy()
                predictions, _, _, features = fit_predict(training, validation, config)
                metrics = score(validation["tiene_evento"], validation["personas_desplazadas"], predictions)
                # La duración permite decidir si un motor cabe en el tiempo de Colab; no entra a promoción.
                folds.append({"fold": i, "validation_rows": len(validation), **metrics, "seconds": round(time.perf_counter() - fold_started, 1)})
                part = validation[KEYS + ["tiene_evento", "personas_desplazadas"]].copy()
                part["fold"] = i
                for name, vector in zip(["prob", "point_log", "q10_log", "q90_log"], predictions, strict=True):
                    part[name] = vector
                oof.append(part)
                dump_json(out / "progress.json", {"finished_folds": folds})
                print(f"{run_id} fold {i}/{len(split_indices)}: AUC={metrics['auc']:.4f} RMSE-log={metrics['rmse_log']:.4f} cobertura={metrics['coverage_80']:.3f} {folds[-1]['seconds']}s", flush=True)
            summary = {f"{metric}_mean": float(np.mean([f[metric] for f in folds])) for metric in ["auc", "rmse_log", "winkler_log", "coverage_80", "puntos_75"]}
            summary["auc_std"] = float(np.std([f["auc"] for f in folds]))
            report = {"summary": summary, "folds": folds}
            if config.get("stress_test", False):
                stress_started = time.perf_counter()
                tr = train["fecha"] <= "2021-09-01"
                va = train["fecha"].between("2021-10-01", "2022-09-01")
                pred, _, _, _ = fit_predict(train.loc[tr], train.loc[va], config)
                report["stress_12_months"] = score(train.loc[va, "tiene_evento"], train.loc[va, "personas_desplazadas"], pred)
                report["stress_12_months"]["seconds"] = round(time.perf_counter() - stress_started, 1)
                print(f"{run_id} stress 12 meses: AUC={report['stress_12_months']['auc']:.4f} cobertura={report['stress_12_months']['coverage_80']:.3f}", flush=True)
                # Horizonte de 13-24 meses: se parece más a la prueba privada (20-39 meses tras el corte).
                tr = train["fecha"] <= "2020-09-01"
                pred, _, _, _ = fit_predict(train.loc[tr], train.loc[va], config)
                report["stress_far_13_24"] = score(train.loc[va, "tiene_evento"], train.loc[va, "personas_desplazadas"], pred)
                print(f"{run_id} stress lejano 13-24 meses: AUC={report['stress_far_13_24']['auc']:.4f} puntos={report['stress_far_13_24']['puntos_75']:.2f}", flush=True)
            dump_json(out / "metrics.json", report)
            pd.concat(oof, ignore_index=True).to_csv(out / "oof.csv", index=False, date_format="%Y-%m-%d")
            pred, feature_model, models, features = fit_predict(train, tests, config)
            submission = tests[KEYS].copy()
            submission["fecha"] = submission["fecha"].dt.strftime("%Y-%m-%d")
            submission[OUTPUT[2]] = pred[0]
            for name, vector in zip(OUTPUT[3:], pred[1:], strict=True):
                submission[name] = np.expm1(vector)
            validate_predictions(submission, tests[KEYS])
            submission.to_csv(out / "predicciones.csv", index=False)
            # Releer el CSV exportado verifica formato/precisión tras serializarlo.
            saved = pd.read_csv(out / "predicciones.csv", dtype={"DIVIPOLA": "string"})
            validate_predictions(saved, tests[KEYS])
            joblib.dump({"features": feature_model, "models": models}, out / "model.joblib")
        from tecton.artifacts import feature_registry, html_dashboard

        feature_registry(features).to_csv(out / "replicabilidad.csv", index=False)
        html_dashboard(submission, report, out / "dashboard.html", synthetic)
        artifact_names = ["source.zip", "metrics.json", "predicciones.csv", "model.joblib", "replicabilidad.csv", "dashboard.html"]
        manifest.update({"status": "complete", "qa_passed": True, "feature_names": features, "elapsed_seconds": time.perf_counter() - started, "artifact_hashes": {name: sha256(out / name) for name in artifact_names}})
        dump_json(out / "manifest.json", manifest)
        print(json.dumps({"run_id": run_id, "synthetic": synthetic, "summary": summary}, ensure_ascii=False), flush=True)
        return run_id
    except BaseException as error:
        manifest.update({"status": "failed", "error_type": type(error).__name__, "elapsed_seconds": time.perf_counter() - started})
        dump_json(out / "manifest.json", manifest)
        raise


def promotion_reasons(candidate, candidate_manifest, current=None, current_manifest=None):
    reasons = []
    if candidate_manifest.get("status") != "complete" or not candidate_manifest.get("qa_passed"):
        reasons.append("Experimento incompleto o sin QA aprobado.")
    if current is None:
        return reasons
    if candidate_manifest["data_hashes"] != current_manifest["data_hashes"]:
        reasons.append("Los datos cambian: no son experimentos comparables.")
    if candidate_manifest["protocol_hash"] != current_manifest["protocol_hash"]:
        reasons.append("Cambió el protocolo de validación.")
    c, old = candidate["summary"], current["summary"]
    # La nota oficial es compuesta: promover por puntos_75 sin sacrificar AUC (50% de la nota).
    if c["puntos_75_mean"] < old["puntos_75_mean"] + 0.2:
        reasons.append("puntos_75 medio no mejora al menos 0.2.")
    if c["auc_mean"] < old["auc_mean"] - 0.002:
        reasons.append("AUC medio baja más de 0.002.")
    improvements = sum(new["puntos_75"] > prev["puntos_75"] for new, prev in zip(candidate["folds"], current["folds"], strict=True))
    if improvements < np.ceil(len(candidate["folds"]) * 0.6):
        reasons.append("No mejora puntos_75 en al menos 60% de los folds.")
    if candidate["folds"][-1]["auc"] < current["folds"][-1]["auc"] - 0.005:
        reasons.append("El último fold pierde más de 0.005 AUC.")
    if c["rmse_log_mean"] > old["rmse_log_mean"] * 1.03:
        reasons.append("RMSE-log empeora más de 3%.")
    if c["winkler_log_mean"] > old["winkler_log_mean"] * 1.10:
        reasons.append("Winkler empeora más de 10%.")
    if "stress_12_months" in candidate and "stress_12_months" in current:
        if candidate["stress_12_months"]["auc"] < current["stress_12_months"]["auc"] - 0.005:
            reasons.append("La prueba de 12 meses pierde más de 0.005 AUC.")
    return reasons


def promote(root: Path, run_id: str, reason: str, demo=False):
    out = resolve_run(root, run_id)
    manifest = json.loads((out / "manifest.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    verify_run_artifacts(out, manifest)
    if manifest["synthetic"] != demo:
        raise ValueError("Usar --demo solo para ensayos sintéticos; nunca promoverlos a producción.")
    if not demo and (len(metrics["folds"]) != 5 or not manifest["config"]["strict"] or "stress_12_months" not in metrics):
        raise ValueError("Champion real requiere cinco folds, auditoría estricta y prueba de 12 meses.")
    if (root / "state" / "FROZEN.json").exists() and not demo:
        raise ValueError("Entrega congelada. Los nuevos experimentos no reemplazan el champion.")
    pointer = root / "state" / ("champion-demo.json" if demo else "champion.json")
    old_metrics = old_manifest = None
    if pointer.exists():
        old_id = json.loads(pointer.read_text())["run_id"]
        old_dir = resolve_run(root, old_id)
        old_metrics = json.loads((old_dir / "metrics.json").read_text())
        old_manifest = json.loads((old_dir / "manifest.json").read_text())
    reasons = promotion_reasons(metrics, manifest, old_metrics, old_manifest)
    if reasons:
        raise ValueError("No promover: " + " ".join(reasons))
    payload = {"run_id": run_id, "reason": reason, "promoted_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics["summary"]}
    dump_json(pointer, payload)
    with (root / "state" / "promotion_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**payload, "demo": demo}, ensure_ascii=False) + "\n")
    return payload


def compare_runs(root: Path, run_ids=None):
    """Resumen agregado de runs completos; mismo formato en CLI y Colab, sin filas."""
    root = Path(root)
    paths = [resolve_run(root, rid) for rid in run_ids] if run_ids else sorted(p.parent for p in (root / "runs").glob("*/metrics.json"))
    rows = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["status"] != "complete":
            continue
        metrics = json.loads((path / "metrics.json").read_text())
        row = {"run_id": path.name, "synthetic": manifest["synthetic"], "protocol_hash": manifest["protocol_hash"], **metrics["summary"]}
        row["folds"] = len(metrics["folds"])
        row["last_fold_auc"] = metrics["folds"][-1]["auc"]
        if "stress_12_months" in metrics:
            row["stress_auc"] = metrics["stress_12_months"]["auc"]
            row["stress_coverage_80"] = metrics["stress_12_months"]["coverage_80"]
            row["stress_puntos"] = metrics["stress_12_months"]["puntos_75"]
        if "stress_far_13_24" in metrics:
            row["far_auc"] = metrics["stress_far_13_24"]["auc"]
            row["far_puntos"] = metrics["stress_far_13_24"]["puntos_75"]
        row["elapsed_seconds"] = round(manifest.get("elapsed_seconds", 0), 1)
        rows.append(row)
    return rows


def resolve_run(root: Path, run_id: str):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("run_id inválido.")
    out = root / "runs" / run_id
    if not out.is_dir():
        raise ValueError(f"Experimento no existe: {run_id}.")
    return out


def verify_run_artifacts(out: Path, manifest):
    hashes = manifest.get("artifact_hashes", {})
    if not {"predicciones.csv", "metrics.json", "source.zip"} <= set(hashes):
        raise ValueError("El run no tiene manifiesto completo de integridad.")
    for name, expected in hashes.items():
        if sha256(out / name) != expected:
            raise ValueError(f"Artefacto cambió después de entrenar: {name}.")
