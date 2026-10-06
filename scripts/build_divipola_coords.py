"""Coordenadas de cabecera municipal (DANE DIVIPOLA, datos.gov.co gdxc-w37w) para vecindad espacial."""

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

URL = "https://www.datos.gov.co/resource/gdxc-w37w.json?$limit=5000"
out = Path("data/external/divipola_coords.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(URL, timeout=120) as response:
    rows = json.load(response)
frame = pd.DataFrame(rows)
frame = pd.DataFrame({
    "DIVIPOLA": frame["cod_mpio"].astype(str).str.zfill(5),
    "lat": frame["latitud"].str.replace(",", ".").astype(float),
    "lon": frame["longitud"].str.replace(",", ".").astype(float),
}).drop_duplicates("DIVIPOLA")
frame.to_csv(out, index=False)
out.with_suffix(".meta.json").write_text(json.dumps({
    "fuente": "DANE DIVIPOLA - Códigos municipios", "dataset_id": "gdxc-w37w", "url": URL,
    "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(), "municipios": len(frame),
    "uso": "Coordenadas de cabecera para vecindad espacial; referencia geográfica estática, sin dimensión temporal.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(out, len(frame), "municipios")
