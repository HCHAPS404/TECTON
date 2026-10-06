"""Historial UNGRD de emergencias climaticas 2014-2017 por municipio.

Fuente: consolidados anuales UNGRD
https://portal.gestiondelriesgo.gov.co/Paginas/Consolidado-Atencion-de-Emergencias.aspx
Ejecutar: uv run --frozen --with xlrd --with openpyxl python scripts/build_ungrd_hist_2014_2017.py
Usa los archivos presentes en data/external/cache/ungrd_hist (descarga los faltantes si --download).
"""
from __future__ import annotations

import json
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/external/cache/ungrd_hist"
OUT = ROOT / "data/external/ungrd_hist_2014_2017.csv"
OUT_MES = ROOT / "data/external/ungrd_hist_2014_2017_mes.csv"
META = ROOT / "data/external/ungrd_hist_2014_2017.meta.json"
BASE = "https://portal.gestiondelriesgo.gov.co/Documents/consolidado-emergencias/"
FILES = {2014: "EMERGENCIAS2014.xls", 2015: "EMERGENCIAS_2015.xls",
         2016: "EMERGENCIAS2016.xls", 2017: "EMERGENCIAS_2017.xls"}
YEARS = list(FILES)

CLIMA_KEYS = ["INUNDACION", "MOVIMIENTO EN MASA", "DESLIZAMIENTO", "VENDAVAL", "CRECIENTE",
              "AVENIDA", "GRANIZ", "TORMENTA", "TEMPORAL", "EROSION", "SEQUIA",
              "INCENDIO FORESTAL", "INCENDIO DE COBERTURA", "HURACAN", "TORNADO",
              "LLUVIA", "AVALANCHA", "REMOCION", "DESBORDAMIENTO", "MAR DE LEVA", "HELADA", "DESABASTECIMIENTO"]
EXCLUDE_KEYS = ["SISMO", "VOLCAN", "ESTRUCTURAL", "ACCIDENTE", "COLAPSO", "EXPLOSION"]


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def is_clima(ev: str) -> bool:
    e = norm(ev)
    if any(k in e for k in EXCLUDE_KEYS):
        return False
    return any(k in e for k in CLIMA_KEYS)


def load_year(year: int) -> pd.DataFrame | None:
    p = CACHE / FILES[year]
    if not p.exists() or p.stat().st_size < 10000:
        return None
    head = p.read_bytes()[:4]
    engine = "openpyxl" if head.startswith(b"PK") else "xlrd"
    try:
        raw = pd.read_excel(p, sheet_name="REPORTE DE EMERGENCIAS", header=None, engine=engine)
    except Exception as exc:  # incomplete download etc.
        print(f"[{year}] no legible: {exc}", file=sys.stderr)
        return None
    hdr = None
    for i in range(min(30, len(raw))):
        vals = [norm(x) for x in raw.iloc[i].tolist()]
        if "FECHA" in vals and "EVENTO" in vals:
            hdr = i
            break
    if hdr is None:
        return None
    cols = [norm(x) for x in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    c_div = next(c for c in cols if c.startswith("CODIFICACION") or "DIVIPOLA" in c)
    out = pd.DataFrame({
        "fecha": pd.to_datetime(df["FECHA"], errors="coerce"),
        "evento": df["EVENTO"].astype(str),
        "DIVIPOLA": pd.to_numeric(df[c_div], errors="coerce"),
        "personas": pd.to_numeric(df.get("PERSONAS"), errors="coerce").fillna(0),
    })
    out = out.dropna(subset=["fecha", "DIVIPOLA"])
    out = out[out["fecha"].dt.year == year]  # solo el anio del archivo
    out["DIVIPOLA"] = out["DIVIPOLA"].astype(int).astype(str).str.zfill(5)
    return out


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if "--download" in sys.argv:
        for y, f in FILES.items():
            if not (CACHE / f).exists():
                req = urllib.request.Request(BASE + f, headers={"User-Agent": "Mozilla/5.0"})
                (CACHE / f).write_bytes(urllib.request.urlopen(req, timeout=600).read())

    frames, per_year = [], {}
    for y in YEARS:
        d = load_year(y)
        if d is not None:
            per_year[y] = {"filas": int(len(d))}
            frames.append(d)
    if not frames:
        raise SystemExit("sin archivos legibles")
    ev = pd.concat(frames, ignore_index=True)
    years_ok = sorted(per_year)
    tipos = ev["evento"].map(norm).value_counts()
    ev["clima"] = ev["evento"].map(is_clima)
    clima = ev[ev["clima"]].copy()
    for y in years_ok:
        per_year[y]["filas_clima"] = int((clima["fecha"].dt.year == y).sum())
    clima["ym"] = clima["fecha"].dt.to_period("M")
    clima["mes"] = clima["fecha"].dt.month
    clima["anio"] = clima["fecha"].dt.year

    div = pd.read_csv(ROOT / "data/external/divipola_coords.csv", dtype={"DIVIPOLA": str})[["DIVIPOLA"]]
    div["DIVIPOLA"] = div["DIVIPOLA"].str.zfill(5)
    n_meses = 12 * len(years_ok)

    g = clima.groupby("DIVIPOLA").agg(hist14_meses_con_evento=("ym", "nunique"),
                                      hist14_n_eventos=("evento", "size"),
                                      afect=("personas", "sum")).reset_index()
    res = div.merge(g, on="DIVIPOLA", how="left").fillna(0)
    res["hist14_meses_con_evento"] = res["hist14_meses_con_evento"].astype(int)
    res["hist14_n_eventos"] = res["hist14_n_eventos"].astype(int)
    res["hist14_tasa"] = res["hist14_meses_con_evento"] / n_meses
    res["hist14_log_afectados"] = np.log1p(res["afect"].clip(lower=0))
    res = res[["DIVIPOLA", "hist14_meses_con_evento", "hist14_tasa", "hist14_n_eventos",
               "hist14_log_afectados"]]
    res.to_csv(OUT, index=False)

    gm = clima.groupby(["DIVIPOLA", "mes"])["anio"].nunique().rename("n").reset_index()
    grid = pd.MultiIndex.from_product([div["DIVIPOLA"], range(1, 13)],
                                      names=["DIVIPOLA", "mes"]).to_frame(index=False)
    m = grid.merge(gm, on=["DIVIPOLA", "mes"], how="left").fillna(0)
    m["hist14_tasa_mes"] = m["n"] / len(years_ok)
    m[["DIVIPOLA", "mes", "hist14_tasa_mes"]].to_csv(OUT_MES, index=False)

    unmatched = sorted(set(clima["DIVIPOLA"]) - set(div["DIVIPOLA"]))
    meta = {
        "fuente": "UNGRD - Consolidado anual de atencion de emergencias",
        "pagina": "https://portal.gestiondelriesgo.gov.co/Paginas/Consolidado-Atencion-de-Emergencias.aspx",
        "urls": {y: BASE + FILES[y] for y in years_ok},
        "anios_usados": years_ok,
        "anios_faltantes": [y for y in YEARS if y not in per_year],
        "denominador_meses": n_meses,
        "nota": "hist14_tasa = meses con evento / (12 * anios usados); hist14_tasa_mes = fraccion de anios usados",
        "filtro_tipos_incluye": CLIMA_KEYS,
        "filtro_tipos_excluye": EXCLUDE_KEYS,
        "tipos_incluidos": sorted(t for t in tipos.index if is_clima(t)),
        "tipos_excluidos": sorted(t for t in tipos.index if not is_clima(t)),
        "filas_por_anio": {str(k): v for k, v in per_year.items()},
        "municipios_divipola": int(len(div)),
        "municipios_con_evento": int((res["hist14_n_eventos"] > 0).sum()),
        "codigos_sin_match_divipola": unmatched[:50],
        "afectados": "columna PERSONAS (personas afectadas) sumada",
        "divipola": "data/external/divipola_coords.csv (gdxc-w37w)",
        "fecha_build_utc": datetime.now(timezone.utc).isoformat(),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: meta[k] for k in ["anios_usados", "filas_por_anio", "municipios_con_evento"]}))


if __name__ == "__main__":
    main()
