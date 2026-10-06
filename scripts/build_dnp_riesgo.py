"""Índice Municipal de Riesgo de Desastres ajustado por Capacidades (IMRC, DNP, edición 2024) + MDM (DNP).

Fuentes públicas colombianas:
  - DNP, IMRC-BASE-DE-DATOS-2024.xlsx (hojas 'Exceso de Lluvias' y 'Deficit de Lluvias').
    Insumos con años 2018-2021 (CNPV 2018, IPM 2018, VA/ingresos/inversión GRD 2016-2019, CMGRD/PMGRD/EMRE 2019-2021)
    y capas de amenaza estáticas (SGC 2015, IDEAM 2010-2016). Sin observaciones >= 2022-10.
  - DNP, Medición del Desempeño Municipal (datos.gov.co nkjx-rsq7), años 2018-2020 (media).
  - DIVIPOLA municipios (datos.gov.co gdxc-w37w) como universo.

Reproducir: uv run --frozen python scripts/build_dnp_riesgo.py
Caché: data/external/cache/dnp/. El servidor colaboracion.dnp.gov.co entrega una cadena TLS incompleta;
si la verificación falla se reintenta sin verificar (solo para descarga de archivos públicos).
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_dane_vulnerabilidad import read_xlsx  # noqa: E402  (lector XLSX con librería estándar)

CACHE = ROOT / "data/external/cache/dnp"
OUT = ROOT / "data/external/dnp_riesgo.csv"
META = ROOT / "data/external/dnp_riesgo.meta.json"
IMRC_URL = (
    "https://colaboracion.dnp.gov.co/CDT/PublishingImages/La_Entidad/Directivos/Direccciones/"
    "ambiente-y-desarrollo-sostenible/Gestion-del-Riesgo-de-Desastres/IMRC-BASE-DE-DATOS-2024.xlsx"
)
IMRC_PAGE = (
    "https://www.dnp.gov.co/LaEntidad_/subdireccion-general-prospectiva-desarrollo-nacional/"
    "direccion-ambiente-desarrollo-sostenible/Paginas/indice-municipal-de-gestion-de-riesgo-ajustado-por-capacidades.aspx"
)
MDM_URL = "https://www.datos.gov.co/resource/nkjx-rsq7.json"
DIVIPOLA_URL = "https://www.datos.gov.co/resource/gdxc-w37w.json?$limit=5000"
MDM_YEARS = ["2018", "2019", "2020"]


def fetch(url: str, dest: Path, tries: int = 4, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    insecure = ssl.create_default_context()
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE
    for attempt in range(tries):
        ctx = None if attempt == 0 else insecure
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 TECTON"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                data = r.read()
            dest.write_bytes(data)
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"reintento {attempt + 1} {url}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"no se pudo descargar {url}")


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def imrc_sheet(path: Path, sheet: str, cols: dict[str, str], cats: dict[str, dict[str, float]], instr_flag: bool = False) -> pd.DataFrame:
    raw = read_xlsx(path, sheet)
    header = [re.sub(r"\s+", " ", str(x or "")).strip() for x in raw.iloc[1]]
    body = raw.iloc[2:].copy()
    body.columns = header
    body = body[body["DIVIPOLA"].notna()]
    out = pd.DataFrame({"DIVIPOLA": body["DIVIPOLA"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)})
    instr = [h for h in header if h.upper().startswith(("CMGRD", "PMGRD", "EMRE"))]
    if instr_flag and instr:
        flag = body[instr].apply(lambda c: c.astype(str).str.strip().str.lower().eq("no reportado")).any(axis=1)
        out["imrc_grd_no_reportado"] = flag.astype(float).values
    for src, dst in cols.items():
        match = [h for h in header if h.lower().startswith(src.lower())]
        if not match:
            raise KeyError(f"{sheet}: columna '{src}' no encontrada")
        if dst in cats:
            out[dst] = body[match[0]].astype(str).str.strip().str.lower().map(cats[dst])
        else:
            col = body[match[0]].astype(str).str.strip().str.lower()
            out[dst] = num(col.mask(col.str.startswith("sin "), "0"))  # 'Sin población expuesta' -> 0
    return out.drop_duplicates("DIVIPOLA").set_index("DIVIPOLA")


EXCESO = {
    "%AA": "imrc_e_pct_area_amenazada",
    "%PE": "imrc_e_pct_poblacion_expuesta",
    "Vulnerabilidad social": "imrc_ipm_vulnerabilidad",
    "Indice de Rieso de Desastres Exceso": "imrc_e_indice_riesgo",
    "Componente socioeconómico": "imrc_e_cap_socioeconomica",
    "Componente financiera": "imrc_e_cap_financiera",
    "No. herramientas": "imrc_n_herramientas_grd",
    "Inversión Promedio GRD": "imrc_inversion_grd_pc",
    "Componente gestión del riesgo": "imrc_e_cap_grd",
    "Indice de Capacidades - Exceso": "imrc_e_indice_capacidades",
    "Grupo de Capacidad -Exceso": "imrc_e_grupo_capacidad",
    "Indice de riesgo de desastres ajustado por capacidades": "imrc_e_ajustado",
    "PMGRD": "imrc_pmgrd_estado",
    "EMRE": "imrc_emre_estado",
}
DEFICIT = {
    "% ASE": "imrc_d_pct_area_sequia_extrema",
    "%ASIF": "imrc_d_pct_area_incendio",
    "Indice de Rieso de Desastres - Sequia": "imrc_d_indice_riesgo_sequia",
    "Indice de Rieso de Desastres Incendio": "imrc_d_indice_riesgo_incendio",
    "Indice de Rieso de Desastres - Deficit": "imrc_d_indice_riesgo",
    "Promedio Componente Gestion MDM": "imrc_d_mdm_gestion_2016_2019",
    "Indice de Capacidades - Deficit": "imrc_d_indice_capacidades",
    "Indice de riesgo de desastres ajustado por capacidades": "imrc_d_ajustado",
}
GRUPO = {"c": 0.0, "g1": 1.0, "g2": 2.0, "g3": 3.0, "g4": 4.0, "g5": 5.0}  # C = ciudades; G1 mayor ... G4 menor capacidad
CATS = {
    "imrc_e_grupo_capacidad": GRUPO,
    "imrc_pmgrd_estado": {"adoptado": 2.0, "formulado": 1.0, "no formulado": 0.0},
    "imrc_emre_estado": {"adoptada": 2.0, "adoptado": 2.0, "formulada": 1.0, "formulado": 1.0, "no formulado": 0.0, "no formulada": 0.0},
}


def load_mdm() -> pd.DataFrame:
    where = "anio in(" + ",".join(f"'{y}'" for y in MDM_YEARS) + ") AND indicador in('MDM','Componente de gestión','Componente de resultados')"
    url = MDM_URL + "?" + urllib.parse.urlencode({"$where": where, "$limit": 50000})
    df = pd.DataFrame(json.loads(fetch(url, CACHE / "mdm_2018_2020.json").read_text()))
    df["DIVIPOLA"] = df["codigo_entidad"].astype(str).str.zfill(5)
    df["dato"] = num(df["dato"])
    name = {"MDM": "mdm_puntaje", "Componente de gestión": "mdm_gestion", "Componente de resultados": "mdm_resultados"}
    df["var"] = df["indicador"].map(name) + "_2018_2020"
    return df.pivot_table(index="DIVIPOLA", columns="var", values="dato", aggfunc="mean")


def main() -> None:
    xlsx = fetch(IMRC_URL, CACHE / "IMRC-BASE-DE-DATOS-2024.xlsx")
    exc = imrc_sheet(xlsx, "Exceso de Lluvias", EXCESO, CATS, instr_flag=True)
    dfc = imrc_sheet(xlsx, "Deficit de Lluvias", DEFICIT, {})
    mdm = load_mdm()
    div = pd.DataFrame(json.loads(fetch(DIVIPOLA_URL, CACHE / "divipola.json").read_text()))
    div["DIVIPOLA"] = div["cod_mpio"].astype(str).str.zfill(5)
    div = div.drop_duplicates("DIVIPOLA").set_index("DIVIPOLA")[["cod_dpto"]]

    df = div.join(exc).join(dfc).join(mdm)
    cols = [c for c in df.columns if c != "cod_dpto"]
    raw_cov = {c: int(df[c].notna().sum()) for c in cols}
    missing_any = df[cols].isna().any(axis=1)
    key_missing = df["imrc_e_ajustado"].isna()
    for c in cols:
        med_d = df.groupby("cod_dpto")[c].transform("median")
        df[c] = df[c].fillna(med_d).fillna(df[c].median())
    df["imputado"] = key_missing.astype(int)
    df["imputado_alguna"] = missing_any.astype(int)
    df = df.drop(columns="cod_dpto").reset_index()
    df["DIVIPOLA"] = df["DIVIPOLA"].astype(str).str.zfill(5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6g")

    meta = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_dnp_riesgo.py",
        "filas": int(len(df)),
        "fuentes": {
            "imrc_2024": {"url": IMRC_URL, "pagina": IMRC_PAGE, "entidad": "DNP - Dirección de Ambiente y Desarrollo Sostenible",
                          "edicion": 2024, "universo": "DIVIPOLA 2022 (1.122 municipios + áreas no municipalizadas)"},
            "mdm": {"url": "https://www.datos.gov.co/d/nkjx-rsq7", "entidad": "DNP - Medición del Desempeño Municipal",
                    "anios": MDM_YEARS},
            "divipola": {"url": "https://www.datos.gov.co/d/gdxc-w37w"},
        },
        "anios_insumos": {
            "amenaza_exceso": "SGC 2015 movimientos en masa; IDEAM 2010 flujos torrenciales; IDEAM-IGAC 2012 e IDEAM 2016 inundaciones (capas estáticas)",
            "exposicion": "Grilla de población DANE CNPV 2018",
            "vulnerabilidad": "IPM municipal censal 2018 (DANE)",
            "capacidades": "VA per cápita 2016-2019, ingresos tributarios y no tributarios 2016-2019, población cabecera 2016-2019, densidad empresarial 2019, inversión GRD 2016-2019, CMGRD/PMGRD/EMRE 2019-2021",
            "deficit": "Escenarios de sequía extrema SPI ('varios años', según DNP), susceptibilidad a incendios IDEAM 2009, coberturas 2012, IUA ENA 2019, vulnerabilidad hídrica 2018, MDM gestión 2016-2019",
            "mdm": "media 2018-2020 (2021 no publicado en nkjx-rsq7)",
        },
        "definiciones": {
            "imrc_e_pct_area_amenazada": "% del área municipal en amenaza alta por movimientos en masa, flujos torrenciales o inundaciones lentas (unión)",
            "imrc_e_pct_poblacion_expuesta": "% de la población (grilla CNPV 2018) en el área amenazada",
            "imrc_ipm_vulnerabilidad": "Vulnerabilidad social: % IPM censal 2018",
            "imrc_e_indice_riesgo": "Índice de riesgo de desastres por exceso de lluvias (amenaza x exposición x vulnerabilidad), escala DNP",
            "imrc_e_cap_socioeconomica": "Componente socioeconómico de capacidades (0-1)",
            "imrc_e_cap_financiera": "Componente financiero de capacidades (0-1)",
            "imrc_e_cap_grd": "Componente de gestión del riesgo de desastres de capacidades (0-1)",
            "imrc_e_indice_capacidades": "Índice de capacidades (0-1, mayor = más capacidad)",
            "imrc_e_grupo_capacidad": "Grupo de capacidad codificado: C (ciudades) -> 0, G1..G4 -> 1..4 (mayor número = menores dotaciones/capacidades)",
            "imrc_e_ajustado": "IMRC exceso de lluvias: índice de riesgo ajustado por capacidades (0-100, mayor = más riesgo)",
            "imrc_n_herramientas_grd": "0/1/2 herramientas (CMGRD y PMGRD/EMRE)",
            "imrc_inversion_grd_pc": "Inversión promedio GRD per cápita 2016-2019 (COP constantes 2019)",
            "imrc_pmgrd_estado": "Plan Municipal de GRD: 2 adoptado, 1 formulado, 0 no formulado",
            "imrc_emre_estado": "Estrategia Municipal de Respuesta: 2 adoptada, 1 formulada, 0 no formulada",
            "imrc_grd_no_reportado": "1 si el municipio no reportó a la UNGRD el estado de CMGRD/PMGRD/EMRE (PMGRD/EMRE se imputan; CMGRD solo distingue Creado/No reportado, por eso no se incluye aparte)",
            "imrc_d_*": "Componentes del IMRC por déficit de lluvias (sequía extrema e incendios forestales)",
            "mdm_*_2018_2020": "Medición de Desempeño Municipal DNP (puntaje, componente de gestión, componente de resultados), media 2018-2020",
            "imputado": "1 si el municipio no tenía IMRC exceso (imrc_e_ajustado) y se imputó con la mediana departamental",
            "imputado_alguna": "1 si alguna columna se imputó (mediana departamental; si el departamento no tiene datos, mediana nacional)",
        },
        "cobertura_antes_imputacion": raw_cov,
        "imputacion": "mediana del departamento por columna; respaldo mediana nacional",
        "notas": [
            "Se usa la edición 2024 del IMRC porque sus insumos son 2018-2021; la edición original (2018) usa insumos 2005-2016 y no se incluye.",
            "Variables estáticas (corte transversal); ningún insumo corresponde a observaciones de 2022-10 en adelante según la ficha de fuentes del DNP.",
            "Los escenarios de sequía del componente de déficit figuran como 'varios años' en la ficha DNP; usar imrc_d_* con esa cautela.",
        ],
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"{OUT}: {df.shape}, imputado={int(df['imputado'].sum())}, imputado_alguna={int(df['imputado_alguna'].sum())}")
    print(json.dumps(raw_cov, indent=1))


if __name__ == "__main__":
    main()
