"""Fuentes públicas de Colombia para el motor econométrico.

Solo entran al modelo con corte temporal:
- DANE, población municipal 2018-2022 (proyección CNPV 2018). Después de 2022 se congela 2022.
- UNGRD datos abiertos, conteos climáticos de 2019-01 a 2022-09. Octubre de 2022 en adelante no se descarga.
  El archivo público no trae 2018; ese año queda en el CSV suministrado, usado solo como rezago.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DANE_URL = "https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/PPED-AreaMun-2018-2042_VP.xlsx"
DANE_PAGE = "https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion"
UNGRD_PAGE = "https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Emergencias-UNGRD-/wwkg-r6te"
UNGRD_API = "https://www.datos.gov.co/resource/wwkg-r6te.json"
UNGRD_END = "2022-10-01T00:00:00"
PUBLIC_CAP = pd.Timestamp("2022-09-01")

CLIMATE = {
    "inundacion": ("INUND", "INMERSION"),
    "movimiento": ("MOVIMIENTO", "EROSION"),
    "vendaval": ("VENDAVAL", "TEMPORAL", "GRANIZ", "TORMENTA"),
    "creciente": ("CRECIENTE",),
    "avenida": ("AVENIDA", "TORREN"),
    "otros_clima": ("SEQUIA", "HELADA", "SISMO", "CICLON", "ONDA TROPICAL", "LLUVIA", "VOLCAN"),
}


def cache_dir(root: Path) -> Path:
    path = Path(root) / "data" / "external"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dane_path(root: Path) -> Path:
    return cache_dir(root) / "dane_poblacion_2018_2022.csv"


def ungrd_path(root: Path) -> Path:
    return cache_dir(root) / "ungrd_clima_2019_2022_09.csv"


def _climate_family(evento: str):
    text = str(evento).upper().translate(str.maketrans("ÁÉÍÓÚ", "AEIOU"))
    for name, keys in CLIMATE.items():
        if any(key in text for key in keys):
            return name
    return None


def build_dane(root: Path) -> pd.DataFrame:
    import openpyxl

    target = cache_dir(root) / "PPED-AreaMun-2018-2042_VP.xlsx"
    if not target.exists():
        urllib.request.urlretrieve(DANE_URL, target)
    book = openpyxl.load_workbook(target, read_only=True, data_only=True)
    sheet = book["PobMunicipalxÁrea"]
    rows = []
    for i, row in enumerate(sheet.iter_rows(values_only=True)):
        if i < 9 or row[5] != "Total" or row[2] is None or row[4] is None:
            continue
        year = int(row[4])
        if 2018 <= year <= 2022:
            rows.append({"DIVIPOLA": str(row[2]).split(".")[0].zfill(5), "anio": year, "poblacion": float(row[6])})
    frame = pd.DataFrame(rows).drop_duplicates(["DIVIPOLA", "anio"])
    if frame.empty or not frame["anio"].between(2018, 2022).all():
        raise ValueError("La extracción DANE no quedó en 2018-2022.")
    frame.to_csv(dane_path(root), index=False)
    return frame


def build_ungrd(root: Path) -> pd.DataFrame:
    """Agrega el registro público. El filtro de fecha excluye desde 2022-10."""
    batches, offset = [], 0
    while True:
        params = {
            "$select": "divipola, date_trunc_ym(fecha) as ym, evento, count(1) as n",
            "$where": f"fecha >= '2019-01-01T00:00:00' AND fecha < '{UNGRD_END}'",
            "$group": "divipola, ym, evento",
            "$limit": 50000,
            "$offset": offset,
        }
        url = UNGRD_API + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=180) as response:
            batch = json.load(response)
        if not batch:
            break
        batches.extend(batch)
        if len(batch) < 50000:
            break
        offset += 50000
    raw = pd.DataFrame(batches)
    raw["fecha"] = pd.to_datetime(raw["ym"]).dt.to_period("M").dt.to_timestamp()
    if raw["fecha"].max() > PUBLIC_CAP:
        raise ValueError("La descarga UNGRD pasó de 2022-09.")
    raw["familia"] = raw["evento"].map(_climate_family)
    raw = raw.dropna(subset=["familia"])
    raw["n"] = pd.to_numeric(raw["n"])
    raw["DIVIPOLA"] = raw["divipola"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    counts = raw.groupby(["DIVIPOLA", "fecha", "familia"], as_index=False)["n"].sum()
    wide = counts.pivot(index=["DIVIPOLA", "fecha"], columns="familia", values="n").fillna(0).reset_index()
    for name in CLIMATE:
        if name not in wide.columns:
            wide[name] = 0.0
    wide = wide[["DIVIPOLA", "fecha", *CLIMATE]]
    wide.to_csv(ungrd_path(root), index=False, date_format="%Y-%m-%d")
    return wide


def load_dane(root: Path) -> pd.DataFrame:
    path = dane_path(root)
    if not path.exists():
        return pd.DataFrame(columns=["DIVIPOLA", "anio", "poblacion"])
    frame = pd.read_csv(path, dtype={"DIVIPOLA": "string"})
    frame["anio"] = frame["anio"].astype(int)
    frame = frame[frame["anio"].between(2018, 2022)]
    return frame


def load_ungrd(root: Path) -> pd.DataFrame:
    path = ungrd_path(root)
    columns = ["DIVIPOLA", "fecha", *CLIMATE]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype={"DIVIPOLA": "string"}, parse_dates=["fecha"])
    frame = frame[frame["fecha"] <= PUBLIC_CAP]
    return frame
