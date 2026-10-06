"""Contrato de datos y formato de entrega. Nunca modifica los CSV de entrada."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["DIVIPOLA", "fecha"]
TARGETS = ["tiene_evento", "personas_desplazadas"]
NUMERIC = [
    "densidad_poblacional", "ingresos_tributarios", "inversion_gestion_riesgo", "nbi_pct",
    "elevacion_media", "pendiente_media", "pct_pendiente_alta", "ONI",
]
COMMON = KEYS + ["anio", "mes", "fase_enso"] + NUMERIC
OUTPUT = KEYS + [
    "prob_evento", "personas_desplazadas_estimadas", "personas_desplazadas_q10",
    "personas_desplazadas_q90",
]
FILES = ["entrenamiento.csv", "prueba_equipos.csv", "prueba_oculta.csv"]
PERIODS = [
    ("2018-01-01", "2022-09-01", 62814),
    ("2022-10-01", "2024-04-01", 20938),
    ("2024-05-01", "2025-12-01", 22040),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(data_dir: Path, strict: bool = True):
    synthetic = (data_dir / "SYNTHETIC.json").exists()
    if not strict and not synthetic:
        raise ValueError("strict=false solo se permite con datos sintéticos marcados.")
    frames, summary, hashes = [], {}, {}
    for i, filename in enumerate(FILES):
        path = data_dir / filename
        frame = pd.read_csv(path, dtype={"DIVIPOLA": "string"})
        required = COMMON + (TARGETS if i == 0 else [])
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{filename}: faltan columnas {missing}. Revisar diccionario oficial.")
        if i and set(TARGETS) & set(frame.columns):
            raise ValueError(f"{filename}: contiene targets; no usarlo como archivo de prueba.")
        if frame["DIVIPOLA"].isna().any() or not frame["DIVIPOLA"].str.fullmatch(r"\d{5}").all():
            raise ValueError(f"{filename}: DIVIPOLA debe conservar exactamente cinco dígitos.")
        frame["fecha"] = pd.to_datetime(frame["fecha"], format="%Y-%m-%d", errors="raise")
        if frame["fecha"].isna().any() or not frame["fecha"].dt.day.eq(1).all():
            raise ValueError(f"{filename}: fecha debe ser AAAA-MM-01.")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{filename}: llaves duplicadas.")
        for col in NUMERIC + ["anio", "mes"] + (TARGETS if i == 0 else []):
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if np.isinf(frame[col].to_numpy(dtype=float)).any():
                raise ValueError(f"{filename}: infinito en {col}.")
        if not frame["anio"].eq(frame["fecha"].dt.year).all() or not frame["mes"].eq(frame["fecha"].dt.month).all():
            raise ValueError(f"{filename}: anio/mes no coincide con fecha.")
        start, end, expected = PERIODS[i]
        if strict:
            months = pd.date_range(start, end, freq="MS")
            if len(frame) != expected or set(frame["fecha"]) != set(months):
                raise ValueError(f"{filename}: filas o períodos no coinciden con la guía.")
            if frame["DIVIPOLA"].nunique() != 1102 or not frame.groupby("fecha").size().eq(1102).all():
                raise ValueError(f"{filename}: se esperan 1102 municipios por mes.")
        if i == 0:
            if frame[TARGETS].isna().any().any():
                raise ValueError("Los targets de entrenamiento no pueden tener nulos.")
            if not frame["tiene_evento"].isin([0, 1]).all():
                raise ValueError("tiene_evento no es binario.")
            people = frame["personas_desplazadas"]
            if (people < 0).any() or not np.equal(people, np.floor(people)).all():
                raise ValueError("personas_desplazadas debe ser entero no negativo.")
            if ((frame["tiene_evento"] == 0) & (people > 0)).any():
                raise ValueError("Hay personas positivas con tiene_evento=0.")
            # evento=1/personas=0 es válido: puede haber afectación de viviendas.
        summary[filename] = {
            "rows": len(frame), "municipalities": int(frame["DIVIPOLA"].nunique()),
            "date_min": str(frame["fecha"].min().date()),
            "date_max": str(frame["fecha"].max().date()),
            "missing_covariates": {c: int(frame[c].isna().sum()) for c in NUMERIC},
        }
        hashes[filename] = sha256(path)
        frames.append(frame)
    if any(set(f["DIVIPOLA"]) != set(frames[0]["DIVIPOLA"]) for f in frames[1:]):
        raise ValueError("Los tres archivos no contienen los mismos municipios.")
    if pd.concat([f[KEYS] for f in frames]).duplicated(KEYS).any():
        raise ValueError("Hay solapamiento de llaves entre archivos.")
    summary["training"] = {
        "event_prevalence": float(frames[0]["tiene_evento"].mean()),
        "zero_people_rate": float(frames[0]["personas_desplazadas"].eq(0).mean()),
    }
    return frames, summary, hashes, synthetic


def validate_predictions(pred: pd.DataFrame, expected_keys: pd.DataFrame) -> None:
    if list(pred.columns) != OUTPUT:
        raise ValueError(f"Columnas deben ser exactamente {OUTPUT}.")
    if len(pred) != len(expected_keys) or pred.duplicated(KEYS).any():
        raise ValueError("Filas faltantes/adicionales o llaves duplicadas.")
    normalized = expected_keys[KEYS].copy()
    normalized["fecha"] = pd.to_datetime(normalized["fecha"]).dt.strftime("%Y-%m-%d")
    actual = pred[KEYS].copy()
    actual["fecha"] = pd.to_datetime(actual["fecha"]).dt.strftime("%Y-%m-%d")
    if not actual.reset_index(drop=True).astype(str).equals(normalized.reset_index(drop=True).astype(str)):
        raise ValueError("Llaves u orden no corresponden a los dos archivos de prueba concatenados.")
    if not pred["DIVIPOLA"].astype(str).str.fullmatch(r"\d{5}").all():
        raise ValueError("DIVIPOLA inválido en entrega.")
    values = pred[OUTPUT[2:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("La entrega contiene nulos o infinitos.")
    if not pred["prob_evento"].between(0, 1).all() or (values[:, 1:] < 0).any():
        raise ValueError("Probabilidad o magnitud fuera de dominio.")
    if (pred["personas_desplazadas_q10"] > pred["personas_desplazadas_q90"]).any():
        raise ValueError("Intervalos cruzados.")
