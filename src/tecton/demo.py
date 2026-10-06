"""Datos inventados para probar código; nunca se presentan como evidencia del reto."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tecton.schema import COMMON, FILES, PERIODS


def generate(out: Path, municipalities=20, seed=42):
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if p.name != ".gitkeep"]
    if existing and not (out / "SYNTHETIC.json").exists():
        raise ValueError("No sobrescribir una carpeta existente sin marcador sintético.")
    rng = np.random.default_rng(seed)
    codes = [f"{(i % 10) + 5:02d}{(i // 10) + 1:03d}" for i in range(municipalities)]
    rows = []
    for i, code in enumerate(codes):
        elevation, slope, nbi, density = rng.uniform(0, 2800), rng.uniform(1, 30), rng.uniform(5, 65), rng.uniform(5, 900)
        fiscal = rng.uniform(50000, 800000)
        investment = rng.uniform(0, 1e8)
        for date in pd.date_range("2018-01-01", "2025-12-01", freq="MS"):
            oni = np.sin((date.year - 2018) * 0.8 + date.month / 7)
            risk = 0.08 + 0.18 * (i % 4) / 3 + 0.05 * (1 + np.sin(date.month * np.pi / 6))
            event = int(rng.uniform() < risk)
            people = max(0, int(event * (rng.uniform() > 0.15) * np.expm1(rng.normal(3.1 + slope / 40, 0.7))))
            rows.append({
                "DIVIPOLA": code, "fecha": date, "anio": date.year, "mes": date.month,
                "densidad_poblacional": density * (1 + 0.01 * (date.year - 2018)),
                "ingresos_tributarios": fiscal, "inversion_gestion_riesgo": investment,
                "nbi_pct": nbi, "elevacion_media": elevation, "pendiente_media": slope,
                "pct_pendiente_alta": slope * 2, "ONI": oni,
                "fase_enso": "El Nino" if oni >= 0.5 else "La Nina" if oni <= -0.5 else "Neutral",
                "tiene_evento": event, "personas_desplazadas": people,
                "n_eventos": event, "familias_desplazadas": people // 4,
                "viv_destruidas": int(event and people == 0), "viv_averiadas": event,
                "n_movimiento_masa": 0, "n_inundacion": event, "n_vendaval": 0,
                "n_creciente_subita": 0, "n_avenida_torrencial": 0, "n_otros": 0,
            })
    frame = pd.DataFrame(rows)
    for i, (filename, period) in enumerate(zip(FILES, PERIODS, strict=True)):
        start, end, _ = period
        part = frame[frame.fecha.between(start, end)].copy()
        if i:
            part = part[COMMON]
        part.to_csv(out / filename, index=False, date_format="%Y-%m-%d")
    (out / "SYNTHETIC.json").write_text(json.dumps({"synthetic": True, "municipalities": municipalities, "seed": seed}), encoding="utf-8")
