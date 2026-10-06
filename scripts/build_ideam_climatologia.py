"""Construye la climatologia mensual de precipitacion IDEAM por municipio (DIVIPOLA).

Fuente: datos.gov.co (Socrata)
  - Precipitacion IDEAM (sensores automaticos/telemetria): s54a-sgyg
  - DIVIPOLA DANE (codigos y coordenadas de cabecera): gdxc-w37w

Ventana temporal estricta: 2018-01-01 00:00:00 .. 2022-09-30 23:59:59 (57 meses).

Pasos:
 1. Para cada mes de la ventana, agregacion del lado del servidor por estacion:
    n (observaciones), mm (suma), max, min, npos (obs > 0). Cache en
    data/external/cache/ideam_s54a/YYYY-MM.json (re-ejecutable, reanudable).
 2. Metadatos de estaciones (nombre, depto, municipio, lat, lon) por anio.
 3. Control de calidad por estacion-mes (ver QC_* abajo) y reescalado por
    completitud: mm_mes = mm * 1/completitud, con completitud = n / (tasa_max*dias),
    tasa_max = maximo de n/dias del mes para la estacion (frecuencia nominal).
 4. Climatologia por estacion y mes calendario = media de los anios validos.
 5. Asignacion estacion -> municipio por nombre normalizado + departamento;
    si no casa, municipio con cabecera mas cercana.
 6. Municipio con estaciones: media de sus estaciones (metodo='estacion').
    Sin estacion: IDW (p=2) de los 3 municipios-con-dato mas cercanos (metodo='vecina').

Uso: uv run --frozen python scripts/build_ideam_climatologia.py
"""
from __future__ import annotations

import calendar
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"
CACHE = OUT_DIR / "cache" / "ideam_s54a"
OUT_CSV = OUT_DIR / "ideam_climatologia.csv"
OUT_META = OUT_DIR / "ideam_climatologia.meta.json"

BASE = "https://www.datos.gov.co/resource/"
PRECIP_ID = "s54a-sgyg"
DIVIPOLA_ID = "gdxc-w37w"
START = (2018, 1)
END = (2022, 9)  # inclusive; ultimo instante 2022-09-30T23:59:59
WORKERS = 4

QC_MIN_COMPLETENESS = 0.70  # fraccion minima de observaciones esperadas
QC_MAX_FRAC_POS = 0.50      # >50% de intervalos con lluvia -> sensor pegado
QC_MAX_INTERVAL_MM = 150.0  # max por intervalo implausible
QC_MAX_MONTH_MM = 2500.0    # mm/mes implausible (tras reescalado)
QC_MIN_YEARS = 1            # anios validos minimos por estacion-mes calendario
QC_MIN_ANNUAL_MM = 200.0    # total anual climatologico minimo (sensor muerto / reporta ceros)
QC_NEIGH_K = 5              # vecinas para la prueba de consistencia espacial
QC_NEIGH_RATIO = (0.2, 5.0) # total anual / mediana de vecinas fuera de este rango -> se descarta
IDW_K = 3
IDW_P = 2.0


def months():
    y, m = START
    while (y, m) <= END:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def soql(dataset: str, params: dict, timeout=600, retries=5) -> list[dict]:
    url = BASE + dataset + ".json?" + urllib.parse.urlencode(params)
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"fallo consulta {dataset}: {last}")


def fetch_month(ym):
    y, m = ym
    path = CACHE / f"{y}-{m:02d}.json"
    if path.exists():
        return path
    last = calendar.monthrange(y, m)[1]
    rows = soql(PRECIP_ID, {
        "$select": "codigoestacion, count(*) as n, sum(valorobservado::number) as mm, "
                   "max(valorobservado::number) as mx, min(valorobservado::number) as mn, "
                   "sum(case(valorobservado::number > 0, 1, true, 0)) as npos",
        "$where": f"fechaobservacion between '{y}-{m:02d}-01T00:00:00' and "
                  f"'{y}-{m:02d}-{last}T23:59:59'",
        "$group": "codigoestacion",
        "$limit": 50000,
    })
    path.write_text(json.dumps(rows))
    print(f"  {y}-{m:02d}: {len(rows)} estaciones", flush=True)
    return path


def fetch_station_meta(code):
    """Metadatos (nombre, depto, municipio, lat, lon) de una estacion: 1 registro dentro de la ventana.
    (Una agregacion global por anio hace timeout en Socrata; la consulta por estacion es ~4 s.)"""
    path = CACHE / "stations" / f"{code}.json"
    if path.exists():
        return path
    rows = soql(PRECIP_ID, {
        "$select": "codigoestacion, nombreestacion, departamento, municipio, latitud, longitud",
        "$where": f"codigoestacion='{code}' and fechaobservacion between "
                  "'2018-01-01T00:00:00' and '2022-09-30T23:59:59'",
        "$limit": 1,
    }, timeout=180)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return path


def norm(s) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\b(D C|DC|DISTRITO CAPITAL)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


DEPTO_ALIAS = {
    "BOGOTA": "BOGOTA", "BOGOTA D C": "BOGOTA",
    "SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES",
    "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES",
    "SAN ANDRES": "SAN ANDRES", "VALLE": "VALLE DEL CAUCA",
    "NORTE DE SANTANDER": "NORTE DE SANTANDER", "GUAJIRA": "LA GUAJIRA",
}


def norm_dep(s) -> str:
    n = norm(s)
    return DEPTO_ALIAS.get(n, n)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def to_float(x):
    return pd.to_numeric(pd.Series(x).astype(str).str.replace(",", ".", regex=False), errors="coerce")


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    download_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1-2. Descargas
    print("Descargando agregados mensuales s54a-sgyg ...", flush=True)
    yms = list(months())
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(fetch_month, yms))

    div_path = CACHE / "divipola_gdxc-w37w.json"
    if not div_path.exists():
        div_path.write_text(json.dumps(soql(DIVIPOLA_ID, {"$limit": 5000})))

    # 3. QC estacion-mes
    frames = []
    for y, m in yms:
        d = pd.DataFrame(json.loads((CACHE / f"{y}-{m:02d}.json").read_text()))
        d["anio"], d["mes"] = y, m
        frames.append(d)
    sm = pd.concat(frames, ignore_index=True)
    for c in ["n", "mm", "mx", "mn", "npos"]:
        sm[c] = pd.to_numeric(sm[c], errors="coerce")
    sm["dias"] = [calendar.monthrange(y, m)[1] for y, m in zip(sm.anio, sm.mes)]
    sm["rate"] = sm.n / sm.dias
    sm["rate_max"] = sm.groupby("codigoestacion").rate.transform(lambda r: r.quantile(0.95))
    sm["compl"] = (sm.n / (sm.rate_max * sm.dias)).clip(upper=1.0)
    sm["frac_pos"] = sm.npos / sm.n
    n_raw = len(sm)
    ok = (
        (sm.compl >= QC_MIN_COMPLETENESS)
        & (sm.frac_pos <= QC_MAX_FRAC_POS)
        & (sm.mx <= QC_MAX_INTERVAL_MM)
        & (sm.mn >= 0)
        & sm.mm.notna()
    )
    sm = sm[ok].copy()
    sm["mm_mes"] = sm.mm / sm.compl
    sm = sm[sm.mm_mes <= QC_MAX_MONTH_MM]
    qc_stats = {"estacion_mes_total": int(n_raw), "estacion_mes_validos": int(len(sm))}

    clim_st = (sm.groupby(["codigoestacion", "mes"])
                 .agg(mm=("mm_mes", "mean"), mx=("mx", "mean"), wet=("frac_pos", "mean"),
                      n_anios=("anio", "nunique")).reset_index())
    clim_st = clim_st[clim_st.n_anios >= QC_MIN_YEARS]
    # exigir los 12 meses por estacion para no sesgar la estacionalidad
    full = clim_st.groupby("codigoestacion").mes.nunique()
    clim_st = clim_st[clim_st.codigoestacion.isin(full[full == 12].index)]

    # metadatos de estaciones (solo las que pasan QC)
    codes = sorted(clim_st.codigoestacion.unique())
    print(f"Descargando metadatos de {len(codes)} estaciones ...", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        paths = list(ex.map(fetch_station_meta, codes))
    st = pd.DataFrame([r for p in paths for r in json.loads(p.read_text())])
    st["lat"] = to_float(st.latitud).values
    st["lon"] = to_float(st.longitud).values
    st = st.dropna(subset=["lat", "lon"]).drop_duplicates("codigoestacion").copy()
    # coordenadas plausibles para Colombia
    st = st[st.lat.between(-5, 16.5) & st.lon.between(-82.5, -66)]

    # DIVIPOLA
    dv = pd.DataFrame(json.loads(div_path.read_text()))
    dv["DIVIPOLA"] = dv.cod_mpio.astype(str).str.zfill(5)
    dv["lat"] = to_float(dv.latitud).values
    dv["lon"] = to_float(dv.longitud).values
    dv["k_dep"] = dv.dpto.map(norm_dep)
    dv["k_mun"] = dv.nom_mpio.map(norm)
    dv = dv.drop_duplicates("DIVIPOLA").reset_index(drop=True)
    assert dv[["lat", "lon"]].notna().all().all()

    # 5. asignacion estacion -> municipio
    key = {(r.k_dep, r.k_mun): r.DIVIPOLA for r in dv.itertuples()}
    st["k_dep"] = st.departamento.map(norm_dep)
    st["k_mun"] = st.municipio.map(norm)
    st["DIVIPOLA"] = [key.get((a, b)) for a, b in zip(st.k_dep, st.k_mun)]
    st["asignacion"] = np.where(st.DIVIPOLA.notna(), "nombre", "cercania")
    miss = st.DIVIPOLA.isna()
    for i in st.index[miss]:
        dist = haversine(st.at[i, "lat"], st.at[i, "lon"], dv.lat.values, dv.lon.values)
        st.at[i, "DIVIPOLA"] = dv.DIVIPOLA.values[int(np.argmin(dist))]
    # sanity: si el casado por nombre esta a >80 km de la cabecera, usar cercania
    dvi = dv.set_index("DIVIPOLA")
    dist_cab = haversine(st.lat.values, st.lon.values,
                         dvi.loc[st.DIVIPOLA, "lat"].values, dvi.loc[st.DIVIPOLA, "lon"].values)
    far = dist_cab > 80
    for i in st.index[far]:
        dist = haversine(st.at[i, "lat"], st.at[i, "lon"], dv.lat.values, dv.lon.values)
        st.at[i, "DIVIPOLA"] = dv.DIVIPOLA.values[int(np.argmin(dist))]
        st.at[i, "asignacion"] = "cercania(nombre>80km)"

    # QC espacial: total anual de la estacion vs mediana de sus K vecinas mas cercanas
    ann = clim_st.groupby("codigoestacion").mm.sum()
    st = st[st.codigoestacion.isin(ann.index)].reset_index(drop=True)
    st["anual"] = st.codigoestacion.map(ann).values
    ratios = []
    for i in range(len(st)):
        d = haversine(st.lat[i], st.lon[i], st.lat.values, st.lon.values)
        d[i] = np.inf
        idx = np.argsort(d)[:QC_NEIGH_K]
        ratios.append(st.anual[i] / max(np.median(st.anual.values[idx]), 1e-6))
    st["ratio_vecinas"] = ratios
    bad = (st.anual < QC_MIN_ANNUAL_MM) | ~st.ratio_vecinas.between(*QC_NEIGH_RATIO)
    qc_stats["estaciones_12_meses"] = int(len(st))
    qc_stats["estaciones_descartadas_qc_espacial"] = int(bad.sum())
    st = st[~bad]
    cs = clim_st.merge(st[["codigoestacion", "DIVIPOLA"]], on="codigoestacion")
    mun = (cs.groupby(["DIVIPOLA", "mes"])
             .agg(precip_media_mm=("mm", "mean"), intensidad_max_mm=("mx", "mean"), frac_lluvia=("wet", "mean"),
                  n_estaciones=("codigoestacion", "nunique"))
             .reset_index())
    mun["metodo"] = "estacion"
    have = sorted(mun.DIVIPOLA.unique())

    # 6. imputacion IDW
    variables = ["precip_media_mm", "intensidad_max_mm", "frac_lluvia"]
    pivots = {v: mun.pivot(index="DIVIPOLA", columns="mes", values=v).loc[have] for v in variables}
    src = dvi.loc[have]
    rows = []
    for code in dv.DIVIPOLA:
        if code in pivots["precip_media_mm"].index:
            continue
        d = haversine(dvi.at[code, "lat"], dvi.at[code, "lon"], src.lat.values, src.lon.values)
        idx = np.argsort(d)[:IDW_K]
        w = 1.0 / np.maximum(d[idx], 1.0) ** IDW_P
        vals = {v: (pivots[v].values[idx] * w[:, None]).sum(0) / w.sum() for v in variables}
        for mes in range(1, 13):
            rows.append({"DIVIPOLA": code, "mes": mes, **{v: vals[v][mes - 1] for v in variables},
                         "n_estaciones": 0, "metodo": "vecina"})
    full_out = pd.concat([mun, pd.DataFrame(rows)], ignore_index=True).sort_values(["DIVIPOLA", "mes"]).reset_index(drop=True)
    # Climatología de intensidad en archivo aparte: no cambia el archivo usado por el champion.
    inten = full_out[["DIVIPOLA", "mes", "intensidad_max_mm", "frac_lluvia"]].copy()
    inten["intensidad_max_mm"] = inten.intensidad_max_mm.round(3)
    inten["frac_lluvia"] = inten.frac_lluvia.round(5)
    assert inten.notna().all().all() and len(inten) == 12 * len(dv)
    inten.to_csv(OUT_DIR / "ideam_intensidad.csv", index=False)
    out = full_out.drop(columns=["intensidad_max_mm", "frac_lluvia"])
    out["precip_media_mm"] = out.precip_media_mm.round(2)
    out = out.sort_values(["DIVIPOLA", "mes"])[
        ["DIVIPOLA", "mes", "precip_media_mm", "n_estaciones", "metodo"]].reset_index(drop=True)

    # verificaciones
    assert out.DIVIPOLA.nunique() == len(dv)
    assert (out.groupby("DIVIPOLA").mes.nunique() == 12).all()
    assert len(out) == 12 * len(dv)
    assert out.notna().all().all() and (out.precip_media_mm >= 0).all()
    assert out.DIVIPOLA.str.fullmatch(r"\d{5}").all()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_est_mun = int(mun.DIVIPOLA.nunique())
    meta = {
        "archivo": str(OUT_CSV.relative_to(ROOT)),
        "fuentes": [
            {"dataset_id": PRECIP_ID, "nombre": "Precipitacion (IDEAM, estaciones automaticas)",
             "url": f"https://www.datos.gov.co/resource/{PRECIP_ID}.json",
             "pagina": f"https://www.datos.gov.co/d/{PRECIP_ID}", "entidad": "IDEAM"},
            {"dataset_id": DIVIPOLA_ID, "nombre": "DIVIPOLA - Codigos municipios (DANE)",
             "url": f"https://www.datos.gov.co/resource/{DIVIPOLA_ID}.json",
             "pagina": f"https://www.datos.gov.co/d/{DIVIPOLA_ID}", "entidad": "DANE",
             "nota": "catalogo de codigos y coordenadas de cabecera (referencia geografica, sin dimension temporal)"},
        ],
        "datasets_evaluados_descartados": {
            "ksew-j3zj": "vista sin columnas expuestas via API (respuesta vacia)",
            "m84s-22dd": "'PRECIPITACIONES 2017-' mismas columnas/filas que s54a-sgyg (vista derivada); sin ventaja",
            "uwim-sqxg": "'Precipitaciones 2017-20' subconjunto sin latitud/longitud; misma granularidad sub-horaria",
        },
        "fecha_descarga_utc": download_ts,
        "rango_fechas_usado": {"inicio": "2018-01-01T00:00:00", "fin": "2022-09-30T23:59:59",
                               "meses": len(yms),
                               "nota": "meses ene-sep promedian 2018-2022 (5 anios); oct-dic promedian 2018-2021 (4 anios)"},
        "agregacion": (
            "Suma servidor (SoQL) de valorobservado por estacion y mes; QC estacion-mes: "
            f"completitud n/(tasa_p95*dias) >= {QC_MIN_COMPLETENESS} (tasa_p95 = percentil 95 de obs/dia de la estacion), "
            f"fraccion de intervalos con lluvia <= {QC_MAX_FRAC_POS} (descarta sensores pegados), "
            f"max por intervalo <= {QC_MAX_INTERVAL_MM} mm, min >= 0; total reescalado por 1/completitud; "
            f"descartado si > {QC_MAX_MONTH_MM} mm/mes. Climatologia estacion = media de anios validos por mes calendario; "
            "solo estaciones con los 12 meses. QC espacial: se descarta la estacion si su total anual < "
            f"{QC_MIN_ANNUAL_MM} mm o si total/mediana de sus {QC_NEIGH_K} vecinas mas cercanas esta fuera de {QC_NEIGH_RATIO} "
            "(sensores muertos/pegados). Municipio = media simple de sus estaciones."),
        "asignacion_estacion_municipio": (
            "nombre de municipio normalizado (sin tildes, mayusculas, sin parentesis) + departamento contra DIVIPOLA; "
            "si no casa o la cabecera casada esta a >80 km, municipio de cabecera mas cercana (haversine)."),
        "imputacion": f"metodo='vecina': IDW (p={IDW_P}) de los {IDW_K} municipios con estacion mas cercanos por cabecera (distancia minima 1 km).",
        "qc": qc_stats,
        "n_estaciones_usadas": int(cs.codigoestacion.nunique()),
        "asignacion_conteo": st.asignacion.value_counts().to_dict(),
        "cobertura": {"municipios_total": int(len(dv)), "municipios_con_estacion": n_est_mun,
                      "municipios_imputados": int(len(dv) - n_est_mun), "filas": int(len(out))},
        "resumen": {"media_nacional_por_mes_mm": out.groupby("mes").precip_media_mm.mean().round(1).to_dict(),
                    "percentiles_mm": out.precip_media_mm.quantile([0, .05, .5, .95, 1]).round(1).to_dict()},
        "script": "scripts/build_ideam_climatologia.py",
        "tiempo_ejecucion_s": round(time.time() - t0, 1),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: meta[k] for k in ["n_estaciones_usadas", "cobertura", "qc", "asignacion_conteo", "resumen"]},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
