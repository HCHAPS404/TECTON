"""Exporta un geovisor estático. Nunca modifica un run ni entrena modelos."""

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from tecton.pipeline import dump_json, resolve_run, verify_run_artifacts
from tecton.schema import KEYS, load_inputs, sha256, validate_predictions

DANE_LAYER = "https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2024/Serv_CapasMGN_2024/FeatureServer/317"


def normalize_geography(geo):
    if geo.get("type") != "FeatureCollection" or not isinstance(geo.get("features"), list):
        raise ValueError("Cartografía debe ser GeoJSON FeatureCollection.")
    if len(geo["features"]) > 3000:
        raise ValueError("Se esperan como máximo 3000 polígonos municipales.")
    if geo.get("crs") and not any(k in json.dumps(geo["crs"]) for k in ["4326", "CRS84"]):
        raise ValueError("Reproyectar la cartografía a longitud/latitud WGS84 (EPSG:4326).")

    def coordinates(values):
        if not isinstance(values, list) or not values:
            raise ValueError("Coordenadas vacías o inválidas.")
        if isinstance(values[0], (float, int)):
            if len(values) < 2 or not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
                raise ValueError("Coordenadas no finitas o inválidas.")
            if abs(values[0]) > 180 or abs(values[1]) > 90:
                raise ValueError("Coordenadas fuera de WGS84; no cargar metros como grados.")
        else:
            for value in values:
                coordinates(value)

    features, seen = [], set()
    for feature in geo["features"]:
        props = {k.lower(): v for k, v in feature.get("properties", {}).items()}
        code = str(props.get("divipola", props.get("mpio_cdpmp", "")))
        if len(code) != 5 or not code.isascii() or not code.isdigit():
            raise ValueError("Cada geometría requiere DIVIPOLA o mpio_cdpmp de cinco dígitos.")
        if code in seen:
            raise ValueError(f"Geometría duplicada: {code}. Disolver por DIVIPOLA antes de cargar.")
        seen.add(code)
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in ["Polygon", "MultiPolygon"]:
            raise ValueError(f"Geometría {code}: se requiere Polygon o MultiPolygon.")
        coordinates(geometry.get("coordinates"))
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            for ring in polygon:
                if len(ring) < 4 or ring[0][:2] != ring[-1][:2]:
                    raise ValueError(f"Anillo sin cerrar o inválido: {code}.")
        features.append({"type": "Feature", "geometry": geometry, "properties": {
            "DIVIPOLA": code, "municipio": str(props.get("municipio", props.get("mpio_cnmbr", code))),
            "departamento": str(props.get("departamento", props.get("dpto_cnmbr", code[:2]))),
        }})
    return {"type": "FeatureCollection", "features": features}


def download_geography(target: Path):
    """Descarga paginada de geometrías DANE; sin targets ni registros de desastres."""
    target = Path(target)
    if target.exists():
        raise ValueError("El archivo geográfico ya existe; usar otra ruta para conservar su versión.")

    def query(params):
        # El geoportal responde 403 «Forbidden Bots» al agente por defecto de urllib.
        request = Request(DANE_LAYER + "/query?" + urlencode(params), headers={"User-Agent": "Mozilla/5.0 (TECTON geovisor)"})
        with urlopen(request, timeout=90) as response:
            content = json.load(response)
        if "error" in content:
            raise ValueError(f"DANE devolvió un error: {content['error']}")
        return content

    expected = query({"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
    features, offset = [], 0
    while offset < expected:
        page = query({"where": "1=1", "outFields": "mpio_cdpmp,mpio_cnmbr,dpto_cnmbr", "outSR": 4326,
                      "f": "geojson", "orderByFields": "OBJECTID", "resultOffset": offset,
                      "resultRecordCount": 250, "returnGeometry": "true", "maxAllowableOffset": 0.003})
        batch = page.get("features", [])
        if not batch:
            raise ValueError("DANE entregó una página vacía antes de completar la descarga.")
        features.extend(batch)
        offset += len(batch)
    if len(features) != expected:
        raise ValueError("La descarga no coincide con el conteo publicado por DANE.")
    geo = normalize_geography({"type": "FeatureCollection", "features": features})
    dump_json(target, geo)
    dump_json(target.with_suffix(".meta.json"), {"source": DANE_LAYER, "version": "MGN2024", "downloaded_at": datetime.now(timezone.utc).isoformat(), "features": len(features), "crs": "EPSG:4326", "max_allowable_offset_degrees": 0.003, "use": "Visualización simplificada; no usar para cálculo de áreas o features del modelo."})
    return target


def _model_sources(root):
    """Fuentes declaradas que entran al modelo, tal como están en docs/fuentes_datos.csv."""
    path = Path(root) / "docs" / "fuentes_datos.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path, dtype=str).fillna("")
    used = table[table["entra_al_modelo"].str.lower().str.startswith("si")]
    return used[["fuente", "entidad", "uso", "periodo"]].to_dict(orient="records")


def export_dashboard(root: Path, run_id=None, geojson=None, out=None, demo=False):
    root = Path(root).resolve()
    if run_id is None:
        pointer = root / "state" / ("champion-demo.json" if demo else "champion.json")
        if not pointer.exists():
            raise ValueError("Promover un champion o indicar --run-id.")
        run_id = json.loads(pointer.read_text())["run_id"]
    run = resolve_run(root, run_id)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "complete" or not manifest.get("qa_passed"):
        raise ValueError("Se requiere un run completo con QA.")
    if manifest["synthetic"] and not demo:
        raise ValueError("Run sintético: añadir --demo para exportar un ensayo identificado.")
    verify_run_artifacts(run, manifest)
    frames, _, hashes, _ = load_inputs(root / manifest["config"]["data_dir"], manifest["config"]["strict"])
    if hashes != manifest["data_hashes"]:
        raise ValueError("Los CSV cambiaron después de entrenar; conservar los originales.")
    pred = pd.read_csv(run / "predicciones.csv", dtype={"DIVIPOLA": "string"})
    validate_predictions(pred, pd.concat([f[KEYS] for f in frames[1:]], ignore_index=True))
    geography, geo_source = None, None
    if geojson:
        geo_path = Path(geojson)
        geo_path = geo_path if geo_path.is_absolute() else root / geo_path
        geography = normalize_geography(json.loads(geo_path.read_text(encoding="utf-8")))
        meta = geo_path.with_suffix(".meta.json")
        info = json.loads(meta.read_text()) if meta.exists() else {}
        geo_source = info.get("original") or info.get("source") or geo_path.name
    built = root / "dashboard/dist"
    if not (built / "index.html").exists():
        raise ValueError("Falta dashboard/dist. En dashboard ejecutar npm ci && npm run build.")
    target = Path(out) if out else root / "delivery" / run_id / "geovisor"
    target = target if target.is_absolute() else root / target
    if target.exists():
        raise ValueError("La carpeta de salida ya existe. Usar --out con otra ruta para conservar la exportación.")
    municipalities = set(pred.DIVIPOLA)
    mapped = {f["properties"]["DIVIPOLA"] for f in geography["features"]} if geography else set()
    payload = {"schema_version": 1, "synthetic": manifest["synthetic"], "metadata": {
        "source": "predicciones.csv", "run_id": run_id, "model_name": manifest["config"].get("name"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions_sha256": sha256(run / "predicciones.csv"),
        "metrics": json.loads((run / "metrics.json").read_text()), "protocol": manifest["protocol"],
        "geography_source": geo_source, "geography_missing_codes": sorted(municipalities - mapped),
        "geography_sha256": sha256(geo_path) if geojson else None,
        "model_sources": _model_sources(root),
    }, "rows": pred.to_dict(orient="records"), "geography": geography}
    # Construir solo después de validar datos, hashes y geometrías.
    shutil.copytree(built, target)
    dump_json(target / "datos.json", payload)
    (target / "LEEME.txt").write_text("Geovisor local TECTON. Ejecutar: python -m http.server 8765 --bind 127.0.0.1 --directory RUTA_A_ESTA_CARPETA\nAbrir http://127.0.0.1:8765. No abrir index.html con file://. Esta carpeta contiene predicciones: conservar y entregar por canales autorizados del reto. datos.json es un snapshot; no se actualiza al entrenar otro modelo. Sin cartografía, cargar un GeoJSON municipal WGS84 con DIVIPOLA.\n", encoding="utf-8")
    return target
