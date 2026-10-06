"""Variables municipales de vulnerabilidad física (DANE, CNPV 2018) para el modelo de riesgo TECTON.

Fuentes (todas DANE, Censo Nacional de Población y Vivienda 2018, públicas):
  - VIVIENDAS_Cuadros_CNPV_2018.XLSX (cuadros 5V, 6V, 7V, 8V municipales)
  - CNPV-2018-NBI.xlsx (hoja Municipios)
  - deficit-hab-2020-anexo-nueva-metodologia.xlsx (hoja Resumen Municipios)
  - PERSONAS_DEMOGRAFICO_Cuadros_CNPV_2018.xlsx (población por área) [si está disponible]
  - DIVIPOLA municipios (datos.gov.co gdxc-w37w) como universo de municipios.

Reproducir: uv run --frozen python scripts/build_dane_vulnerabilidad.py
Los archivos se cachean en data/external/cache/dane_vuln/. Lector XLSX con la librería estándar
(el entorno congelado no incluye openpyxl).
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/external/cache/dane_vuln"
OUT = ROOT / "data/external/dane_vulnerabilidad.csv"
DANE = "https://www.dane.gov.co/files"
SOURCES = {
    "viviendas": f"{DANE}/censo2018/informacion-tecnica/VIVIENDAS_Cuadros_CNPV_2018.XLSX",
    "nbi": f"{DANE}/censo2018/informacion-tecnica/CNPV-2018-NBI.xlsx",
    "deficit": f"{DANE}/investigaciones/deficit-habitacional/deficit-hab-2020-anexo-nueva-metodologia.xlsx",
    "personas": f"{DANE}/censo2018/informacion-tecnica/PERSONAS_DEMOGRAFICO_Cuadros_CNPV_2018.xlsx",
}
MGN_URL = "https://services.arcgis.com/wLfHepIACaM0pwj9/arcgis/rest/services/MGN_MPIO_POLITICO_2024/FeatureServer/0"
DIVIPOLA_URL = "https://www.datos.gov.co/resource/gdxc-w37w.json?$limit=5000"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def fetch(url: str, dest: Path, tries: int = 4, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 TECTON"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"reintento {attempt + 1} {url}: {exc}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"no se pudo descargar {url}")


def read_xlsx(path: Path, sheet: str) -> pd.DataFrame:
    """Lee una hoja XLSX (valores, sin estilos) como DataFrame sin encabezado."""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.get("Id"): r.get("Target") for r in rels}
        target = None
        for s in wb.find("m:sheets", NS):
            if s.get("name") == sheet:
                target = rel_map[s.get(RNS)]
        if target is None:
            raise KeyError(sheet)
        target = target.lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows: dict[int, dict[int, object]] = {}
        with z.open(target) as fh:
            for _, el in ET.iterparse(fh):
                if el.tag != f"{{{NS['m']}}}row":
                    continue
                r = int(el.get("r")) - 1
                vals: dict[int, object] = {}
                for c in el.findall("m:c", NS):
                    ref = re.match(r"([A-Z]+)", c.get("r")).group(1)
                    col = 0
                    for ch in ref:
                        col = col * 26 + ord(ch) - 64
                    t = c.get("t")
                    v = c.find("m:v", NS)
                    if t == "s" and v is not None:
                        val: object = shared[int(v.text)]
                    elif t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
                    elif v is not None and v.text is not None:
                        try:
                            val = float(v.text)
                        except ValueError:
                            val = v.text
                    else:
                        continue
                    vals[col - 1] = val
                rows[r] = vals
                el.clear()
    nrows = max(rows) + 1
    ncols = max((max(v) for v in rows.values() if v), default=0) + 1
    arr = np.full((nrows, ncols), None, dtype=object)
    for r, vals in rows.items():
        for c, val in vals.items():
            arr[r, c] = val
    return pd.DataFrame(arr)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def code_prefix(s: pd.Series, width: int) -> pd.Series:
    return s.astype(str).str.extract(r"^(\d+)")[0].str.zfill(width)


def cuadro_mpio(df: pd.DataFrame) -> pd.DataFrame:
    """Añade DIVIPOLA a cuadros V_MPIO ('05_Antioquia', '001_Medellín' con forward-fill)."""
    df = df.copy()
    is_code = df[1].astype(str).str.match(r"^\d{3}_")
    dpto = df[0].where(df[0].astype(str).str.match(r"^\d{2}_")).ffill()
    mpio = df[1].where(is_code).ffill()
    df["DIVIPOLA"] = code_prefix(dpto, 2) + code_prefix(mpio, 3)
    df.loc[dpto.isna() | mpio.isna(), "DIVIPOLA"] = None
    df["_start"] = is_code
    return df


def municipal_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Primera fila de cada municipio (la fila de total municipal)."""
    return df[df["_start"]].drop_duplicates("DIVIPOLA").set_index("DIVIPOLA")


def mgn_area_km2(page: int = 100) -> pd.Series:
    cache = CACHE / "mgn2024_area_km2.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"DIVIPOLA": str}).set_index("DIVIPOLA")["area_km2"]
    areas: dict[str, float] = {}
    offset = 0
    while True:
        url = (f"{MGN_URL}/query?where=1%3D1&outFields=MPIO_CDPMP&returnGeometry=true&outSR=102033"
               f"&maxAllowableOffset=50&orderByFields=FID&resultOffset={offset}&resultRecordCount={page}&f=json")
        dest = CACHE / f"_mgn_page_{offset}.json"
        fetch(url, dest, timeout=180)
        data = json.loads(dest.read_text())
        feats = data.get("features", [])
        for f in feats:
            rings = f.get("geometry", {}).get("rings", [])
            a = 0.0
            for r in rings:
                a -= 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1]))
            code = str(f["attributes"]["MPIO_CDPMP"]).zfill(5)
            areas[code] = areas.get(code, 0.0) + a / 1e6
        if len(feats) < page:
            break
        offset += page
    s = pd.Series(areas, name="area_km2").rename_axis("DIVIPOLA")
    s.reset_index().to_csv(cache, index=False)
    for p in CACHE.glob("_mgn_page_*.json"):
        p.unlink()
    return s


def build() -> tuple[pd.DataFrame, dict]:
    files = {k: CACHE / Path(u).name for k, u in SOURCES.items()}
    for k, u in SOURCES.items():
        try:
            fetch(u, files[k])
        except RuntimeError as exc:
            print(exc)
    feats: list[pd.DataFrame] = []
    notes: dict[str, str] = {}

    viv = files["viviendas"]
    # 5V: viviendas ocupadas por número de hogares
    t5 = municipal_total_rows(cuadro_mpio(read_xlsx(viv, "5V_MPIO")))
    occ = num(t5[3])
    h = pd.DataFrame({c: num(t5[i]) for i, c in zip(range(4, 10), [1, 2, 3, 4, 5, 6])})
    hogares = (h * pd.Series({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6})).sum(axis=1)
    feats.append(pd.DataFrame({
        "viviendas_ocupadas_2018": occ,
        "hogares_por_vivienda": hogares / occ,
        "pct_viviendas_multihogar": 100 * (occ - h[1]) / occ,
    }))
    notes["hogares_por_vivienda"] = "Cuadro 5V: hogares/viviendas ocupadas con personas presentes ('6 y más' contado como 6)."

    # 6V: material de paredes exteriores (total municipal, todas las áreas)
    t6 = municipal_total_rows(cuadro_mpio(read_xlsx(viv, "6V_MPIO")))
    tot = num(t6[3])
    col = {n: num(t6[i]) for i, n in enumerate(
        ["bloque", "concreto", "prefabricado", "guadua", "tapia_bahareque_adobe", "madera_burda",
         "cana_esterilla", "desecho", "sin_paredes"], start=4)}
    precario = col["guadua"] + col["madera_burda"] + col["cana_esterilla"] + col["desecho"] + col["sin_paredes"]
    feats.append(pd.DataFrame({
        "pct_paredes_precarias": 100 * precario / tot,
        "pct_paredes_tapia_bahareque_adobe": 100 * col["tapia_bahareque_adobe"] / tot,
        "pct_paredes_madera_burda": 100 * col["madera_burda"] / tot,
        "pct_paredes_desecho_sin_paredes": 100 * (col["desecho"] + col["sin_paredes"]) / tot,
    }))
    notes["pct_paredes_precarias"] = (
        "Cuadro 6V: % viviendas ocupadas con paredes exteriores en guadua; madera burda/tabla/tablón; "
        "caña/esterilla/otros vegetales; materiales de desecho (zinc, tela, cartón, latas, plásticos); sin paredes.")
    notes["pct_paredes_tapia_bahareque_adobe"] = "Cuadro 6V: % con tapia pisada, bahareque o adobe (vulnerabilidad sísmica)."

    # 7V: material de pisos
    t7 = municipal_total_rows(cuadro_mpio(read_xlsx(viv, "7V_MPIO")))
    tot = num(t7[4])
    feats.append(pd.DataFrame({
        "pct_pisos_tierra": 100 * num(t7[10]) / tot,
        "pct_pisos_madera_burda": 100 * num(t7[9]) / tot,
    }))
    notes["pct_pisos_tierra"] = "Cuadro 7V: % viviendas ocupadas con piso de tierra, arena o barro."

    # 8V: servicios públicos
    t8 = municipal_total_rows(cuadro_mpio(read_xlsx(viv, "8V_MPIO")))
    tot = num(t8[3])
    feats.append(pd.DataFrame({
        "pct_sin_energia": 100 * num(t8[6]) / tot,
        "pct_acueducto": 100 * num(t8[8]) / tot,
        "pct_alcantarillado": 100 * num(t8[11]) / tot,
        "pct_recoleccion_basuras": 100 * num(t8[18]) / tot,
        "pct_internet": 100 * num(t8[21]) / tot,
    }))
    notes["pct_acueducto"] = "Cuadro 8V: % viviendas ocupadas con servicio de acueducto."
    notes["pct_alcantarillado"] = "Cuadro 8V: % viviendas ocupadas con servicio de alcantarillado."

    # Población por área: cuadro 8V (cabecera vs centro poblado + rural disperso) en viviendas no trae
    # personas; se usa el cuadro de viviendas ocupadas por área como respaldo.
    d8 = cuadro_mpio(read_xlsx(viv, "8V_MPIO"))
    d8 = d8[d8["DIVIPOLA"].notna()]
    cab = d8[d8[2].astype(str).str.strip().str.lower() == "cabecera"].drop_duplicates("DIVIPOLA").set_index("DIVIPOLA")
    feats.append(pd.DataFrame({"pct_viviendas_rurales": 100 * (1 - num(cab[3]).reindex(tot.index).fillna(0) / tot)}))
    notes["pct_viviendas_rurales"] = "Cuadro 8V: % viviendas ocupadas fuera de la cabecera (centros poblados + rural disperso)."

    # NBI
    nbi = read_xlsx(files["nbi"], "Municipios")
    nbi = nbi[nbi[2].astype(str).str.fullmatch(r"\d{3}(\.0)?")].copy()
    nbi.index = code_prefix(nbi[0], 2) + code_prefix(nbi[2].astype(str).str.replace(".0", "", regex=False), 3)
    feats.append(pd.DataFrame({
        "nbi_pct": num(nbi[4]), "miseria_pct": num(nbi[5]), "nbi_vivienda_pct": num(nbi[6]),
        "nbi_servicios_pct": num(nbi[7]), "nbi_hacinamiento_pct": num(nbi[8]),
        "nbi_rural_pct": num(nbi[18]),
    }))
    notes["nbi_vivienda_pct"] = "NBI componente vivienda: % personas en viviendas inadecuadas (pisos de tierra / paredes precarias)."
    notes["nbi_hacinamiento_pct"] = "NBI componente hacinamiento crítico: % personas en hogares con >3 personas por cuarto."

    # Déficit habitacional (metodología 2020 sobre CNPV 2018)
    dfc = read_xlsx(files["deficit"], "Resumen Municipios")
    dfc = dfc[dfc[2].astype(str).str.fullmatch(r"\d{5}(\.0)?")].copy()
    dfc.index = dfc[2].astype(str).str.replace(".0", "", regex=False).str.zfill(5)
    feats.append(pd.DataFrame({
        "deficit_cuantitativo_pct": num(dfc[4]), "deficit_cualitativo_pct": num(dfc[5]),
        "deficit_habitacional_pct": num(dfc[6]),
    }))
    notes["deficit_habitacional_pct"] = "% hogares en déficit habitacional (cuantitativo + cualitativo), metodología DANE 2020 sobre CNPV 2018."

    # Población censada por área y grupos de edad (cuadro 1PM)
    if files["personas"].exists():
        p = read_xlsx(files["personas"], "1PM")
        mp = p[1].where(p[1].astype(str).str.match(r"^\d{5}_")).ffill()
        p["DIVIPOLA"] = code_prefix(mp, 5)
        p["area"] = p[2].where(p[2].notna()).ffill().astype(str).str.strip().str.lower()
        p = p[mp.notna()]
        p["pers"] = num(p[4])
        tot_rows = p[p[3].astype(str).str.strip() == "Total"]
        pa = tot_rows.pivot_table(index="DIVIPOLA", columns="area", values="pers", aggfunc="first")
        ages = p[(p["area"] == "total") & (p[3].astype(str).str.strip() != "Total")]
        ages = ages.assign(edad=ages[3].astype(str).str.strip())
        ag = ages.pivot_table(index="DIVIPOLA", columns="edad", values="pers", aggfunc="first")
        old = [c for c in ag.columns if c in {"65 a 69", "70 a 74", "75 a 79", "80 a 84", "85 y más"} or c.startswith(("90", "95", "100"))]
        pt = pa["total"]
        feats.append(pd.DataFrame({
            "poblacion_2018": pt,
            "pct_pob_rural": 100 * (pt - pa["cabecera"].fillna(0)) / pt,
            "pct_pob_rural_disperso": 100 * pa["rural disperso"].fillna(0) / pt,
            "pct_pob_0a4": 100 * ag.get("0 a 4") / pt,
            "pct_pob_65mas": 100 * ag[old].sum(axis=1) / pt,
            "personas_por_vivienda": pt,  # se divide abajo por viviendas ocupadas
        }))
        notes["poblacion_2018"] = "Cuadro 1PM: población total censada (CNPV 2018, sin ajuste por cobertura)."
        notes["pct_pob_rural"] = "Cuadro 1PM: % población en resto (centros poblados + rural disperso) = 1 - cabecera/total."
        notes["pct_pob_65mas"] = "Cuadro 1PM: % población de 65 años y más."
        notes["personas_por_vivienda"] = "Población censada / viviendas ocupadas con personas presentes (cuadro 5V)."
    else:
        print("aviso: PERSONAS_DEMOGRAFICO no disponible; se omiten variables de población")

    # Área municipal (km²) a partir de los polígonos MGN DANE 2024 (servicio público UPRA)
    try:
        area = mgn_area_km2()
        feats.append(pd.DataFrame({"area_km2": area}))
        notes["area_km2"] = (
            "Área calculada (km²) de los polígonos MGN_MPIO_POLITICO 2024 (DANE) publicados por UPRA, proyectados a "
            "South America Albers Equal Area (ESRI:102033), generalización 50 m; diferencia típica <1% frente al área oficial.")
    except Exception as exc:  # noqa: BLE001
        print("aviso: área MGN no disponible:", exc)

    out = pd.concat([f[~f.index.duplicated()] for f in feats], axis=1)
    if "area_km2" in out and "poblacion_2018" in out:
        out["densidad_hab_km2"] = out["poblacion_2018"] / out["area_km2"]
        notes["densidad_hab_km2"] = "poblacion_2018 / area_km2."
    if "personas_por_vivienda" in out:
        out["personas_por_vivienda"] = out["personas_por_vivienda"] / out["viviendas_ocupadas_2018"]
    return out, notes


def main() -> None:
    feats, notes = build()
    with urllib.request.urlopen(DIVIPOLA_URL, timeout=120) as r:
        div = pd.DataFrame(json.load(r))
    div["DIVIPOLA"] = div["cod_mpio"].astype(str).str.zfill(5)
    div = div.drop_duplicates("DIVIPOLA")[["DIVIPOLA"]]
    frame = div.merge(feats, left_on="DIVIPOLA", right_index=True, how="left")
    cols = [c for c in frame.columns if c != "DIVIPOLA"]
    frame["imputado"] = frame[cols].isna().any(axis=1).astype(int)
    dpto = frame["DIVIPOLA"].str[:2]
    for c in cols:
        frame[c] = frame[c].fillna(frame.groupby(dpto)[c].transform("median")).fillna(frame[c].median())
    frame[cols] = frame[cols].round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    missing = {c: int(div.merge(feats[[c]], left_on="DIVIPOLA", right_index=True, how="left")[c].isna().sum()) for c in cols}
    meta = {
        "fuente": "DANE - Censo Nacional de Población y Vivienda 2018 (CNPV 2018) y derivados oficiales",
        "urls": SOURCES | {"divipola": DIVIPOLA_URL, "mgn_2024_upra": MGN_URL},
        "dataset_ids": {"divipola_datos_gov_co": "gdxc-w37w"},
        "fecha_referencia_datos": "2018 (CNPV 2018; déficit habitacional publicado 2020 con datos CNPV 2018)",
        "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(),
        "municipios": int(len(frame)),
        "municipios_con_dato_propio_completo": int((frame["imputado"] == 0).sum()),
        "municipios_imputados": int(frame["imputado"].sum()),
        "faltantes_por_variable_antes_de_imputar": missing,
        "imputacion": "Mediana del departamento (2 primeros dígitos DIVIPOLA); si no hay, mediana nacional. imputado=1 si alguna variable fue imputada.",
        "definiciones": notes,
        "uso": "Variables estáticas municipales (sin dimensión temporal); unir por DIVIPOLA.",
        "limitaciones": [
            "Censo 2018 sin ajuste por cobertura (población censada); hay municipios con baja cobertura censal.",
            "27493 (Nuevo Belén de Bajirá) no existía en el CNPV 2018: variables censales imputadas con mediana de Chocó; su área sí proviene del MGN 2024.",
            "area_km2 es geométrica (MGN 2024 vía UPRA, ArcGIS Online) y no el campo oficial MPIO_NAREA; el servicio DANE portalgis.dane.gov.co respondía error durante la descarga.",
            "Porcentajes de materiales/servicios sobre viviendas ocupadas con personas presentes; NBI sobre personas; déficit sobre hogares.",
        ],
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT, frame.shape, "imputados:", int(frame["imputado"].sum()))
    print(json.dumps(missing, indent=1))


if __name__ == "__main__":
    main()
