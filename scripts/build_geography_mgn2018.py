"""Cartografía municipal de respaldo para el geovisor: DANE MGN 2018 (capa MPIO_POLITICO).

Se usa cuando el geoportal DANE (MGN2024) no responde. Solo visualización: no alimenta el modelo.
Uso: uv run --frozen python scripts/build_geography_mgn2018.py
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tecton.dashboard import normalize_geography

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_MPIO_POLITICO.geojson"
TARGET = ROOT / "data" / "geography" / "municipios.geojson"


def main():
    if TARGET.exists():
        raise SystemExit(f"{TARGET} ya existe; borrarlo a mano para regenerar.")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (TECTON geovisor)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    source = json.loads(raw)
    features = [{
        "type": "Feature",
        "geometry": feature["geometry"],
        "properties": {
            "DIVIPOLA": str(feature["properties"]["MPIO_CCNCT"]).zfill(5),
            "municipio": str(feature["properties"]["MPIO_CNMBR"]).title(),
            "departamento": str(feature["properties"]["DPTO_CNMBR"]).title(),
        },
    } for feature in source["features"]]
    geo = normalize_geography({"type": "FeatureCollection", "features": features})
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(geo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    meta = {
        "source": URL,
        "original": "DANE, Marco Geoestadístico Nacional 2018, capa MPIO_POLITICO (réplica pública en GitHub)",
        "version": "MGN2018",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "features": len(geo["features"]),
        "crs": "EPSG:4326",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "use": "Visualización; no usar para cálculo de áreas ni features del modelo.",
    }
    TARGET.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["version", "features", "crs"]}))


if __name__ == "__main__":
    main()
