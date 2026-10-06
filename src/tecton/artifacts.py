import base64
import hashlib
import html
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tecton.pipeline import compare_runs, resolve_run, verify_run_artifacts
from tecton.schema import KEYS, load_inputs, sha256, validate_predictions

# Declaraciones que viajan con el código para que Colab y el bundle las incluyan.
DECLARATIONS = ["docs/ai_usage.csv", "docs/fuentes_datos.csv"]
# Solo agregados: nunca predicciones, OOF, modelos ni CSV oficiales.
REVIEW_FILES = ["manifest.json", "metrics.json", "audit.json", "config.json", "progress.json", "replicabilidad.csv"]


def snapshot(root: Path, target: Path):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        paths = list((root / "src").rglob("*.py")) + list((root / "configs").glob("*.json"))
        paths += [root / name for name in ["pyproject.toml", "uv.lock", "requirements-colab.txt", *DECLARATIONS]]
        # Fuentes externas públicas ya agregadas (no datos del reto) para que Colab reproduzca el modelo.
        paths += list((root / "data" / "external").glob("*.csv")) + list((root / "data" / "external").glob("*.meta.json"))
        for path in sorted(paths):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(root))


def feature_registry(features):
    rows = []
    for name in features:
        source, proxy, limit = "Derivada del calendario", "Calendario gregoriano (universal)", "Reproducir la misma transformación."
        if name in ["DIVIPOLA", "departamento"]:
            source, proxy, limit = "DANE / derivada de DIVIPOLA", "geoBoundaries / código administrativo local", "Los niveles administrativos no son idénticos entre países."
        elif "densidad" in name:
            source, proxy, limit = "TerriData DNP", "WorldPop o GHS-POP (JRC, derivado de satélite) agregado al área administrativa", "Alinear año, unidad y límites administrativos."
        elif "ingresos" in name:
            source, proxy, limit = "TerriData DNP (per cápita)", "Luminosidad nocturna VIIRS/Black Marble (NASA) per cápita como proxy de capacidad económica y fiscal", "VIIRS no mide recaudo; es un proxy correlacionado. Usar finanzas subnacionales oficiales del país si existen."
        elif "inversion" in name:
            source, proxy, limit = "TerriData DNP (total; valor 2020 repetido hasta 2025)", "INFORM Risk Index (JRC), dimensión de falta de capacidad de afrontamiento", "INFORM es nacional o subnacional según país; no mide gasto municipal directo."
        elif name == "nbi_pct":
            source, proxy, limit = "TerriData / Censo DANE 2018", "Encuestas DHS (índice de riqueza), Global MPI subnacional (OPHI) o Relative Wealth Index (satélite)", "Proxy de privación: no equivale exactamente a NBI; revisar resolución y cobertura."
        elif any(word in name for word in ["elevacion", "pendiente"]):
            source, proxy, limit = "Copernicus DEM + agregación municipal", "Copernicus DEM GLO-90 global (misma fuente del dataset)", "Para oni_pendiente usar también NOAA ONI; mantener resolución/unidades."
        elif name in ["ONI", "fase_enso"]:
            source, proxy, limit = "NOAA CPC", "Misma serie ONI de NOAA", "Señal compartida nacional; no mide lluvia local."
        elif name.startswith("ext_ideam"):
            source, proxy, limit = "IDEAM, estaciones automáticas (datos.gov.co s54a-sgyg), climatología 2018-01 a 2022-09", "CHIRPS o ERA5-Land: precipitación mensual media por unidad administrativa", "71% de municipios imputados desde estaciones vecinas; climatología fija, no lluvia observada del mes."
        elif name.startswith("ext_divipola_coords"):
            source, proxy, limit = "DANE DIVIPOLA (datos.gov.co gdxc-w37w), coordenadas de cabecera", "geoBoundaries / GADM: centroide de la unidad administrativa", "Cabecera municipal, no centroide del territorio."
        elif name.startswith("history_spatial"):
            source, proxy, limit = "Targets del entrenamiento de los 8 municipios vecinos (DANE coordenadas), con corte temporal", "DesInventar Sendai agregado por vecindad geográfica", "Depende de la densidad de unidades administrativas del país."
        elif name == "horizonte_meses":
            source, proxy, limit = "Derivada: meses desde el corte", "Calendario", "Reproducir la misma transformación."
        elif name.startswith("history_"):
            source, proxy, limit = "Targets del entrenamiento desde 2018, con corte temporal", "DesInventar Sendai (registro municipal de desastres), EM-DAT o GDACS", "EM-DAT y GDACS no son municipales-mensuales; DesInventar varía en cobertura por país."
        rows.append({"variable": name, "fuente": source, "proxy_global_o_transferible": proxy, "limite": limit})
    return pd.DataFrame(rows)


def html_dashboard(pred, report, target, synthetic=False):
    # HTML sin servidores ni llamadas externas. Datos y predicciones permanecen en el archivo.
    ordered = pred.sort_values("prob_evento", ascending=False).head(40).copy()
    label = "ENSAYO CON DATOS INVENTADOS" if synthetic else "Predicciones PNUD · uso interno"
    metrics = html.escape(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    table = ordered.to_html(index=False, float_format=lambda v: f"{v:.3f}", escape=True)
    content = f"""<!doctype html><html lang="es"><meta charset="utf-8"><title>TECTON PNUD</title>
<style>body{{font:16px system-ui;margin:32px;color:#173042;background:#f8fafb}}h1{{font-size:28px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:right;padding:8px;border-bottom:1px solid #ddd}}
th{{background:#ddecf0}}pre{{padding:18px;background:#eaf0f3}}input{{padding:10px;width:350px;max-width:90%}}
.note{{max-width:1000px;line-height:1.5}}</style><h1>{label}</h1>
<p class="note">Riesgo, magnitud e intervalo predictivo 80%. El target de personas procede de personas reportadas como afectadas según la guía. Los intervalos no garantizan cobertura para cada municipio. No se publican automáticamente los resultados.</p>
<p>{len(pred):,} predicciones · AUC y errores calculados en validación temporal</p><pre>{metrics}</pre>
<h2>40 combinaciones municipio-mes con mayor riesgo</h2><input id="search" placeholder="Filtrar tabla por DIVIPOLA o fecha">
{table}<script>document.getElementById('search').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none');}});</script>
<p>Dashboard básico de respaldo. Puede ampliarse con un mapa sin cambiar el pipeline de predicción.</p></html>"""
    target.write_text(content, encoding="utf-8")


def portable_notebook(source_zip: Path, config: dict, target: Path, data_url: str = ""):
    """Notebook de Colab con el motor embebido. Solo datos oficiales; sin modo sintético.

    data_url solo se inyecta en copias no versionadas para el equipo (el repo es público).
    """
    payload = base64.b64encode(source_zip.read_bytes()).decode()
    digest = hashlib.sha256(source_zip.read_bytes()).hexdigest()
    config = {**config, "data_dir": "data/raw"}
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": f"tecton-{len(cells):02d}"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# TECTON PNUD — entrenamiento y validación en Colab
Motor embebido y verificado por SHA256; llama al mismo código del repositorio, no reimplementa el modelo. CPU, Python 3.10–3.13.

**Uso:** Entorno de ejecución → Ejecutar todo. Para otro experimento, cambiar `CONFIG_NAME` y ejecutar desde **Configuración**; los datos ya cargados se reutilizan.

Reglas del reto: no compartir los datos fuera del equipo, no editar predicciones a mano, entregar solo por el formulario oficial antes del cierre.
""")
    add("code", """from pathlib import Path

# Configuración: None = la embebida (recomendada). Otras: 'first_submission', 'baseline', 'baseline-cal', 'catboost', 'catboost-cal'.
CONFIG_NAME = None
# Hilos de CPU; None conserva los de la configuración (2, como Colab gratuito).
THREADS = None
# Enlace de Drive a hackathon_datos.zip del PNUD. No publicar esta copia del notebook.
URL_DATOS = """ + repr(data_url) + """
# True guarda runs/ y state/ en el Drive del equipo para sobrevivir desconexiones.
USE_DRIVE = False
DRIVE_FOLDER = 'TECTON_PNUD'
""")
    add("code", """import base64, hashlib, io, json, shutil, sys, zipfile

try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if USE_DRIVE and IN_COLAB:
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive') / DRIVE_FOLDER / 'tecton_colab'
else:
    ROOT = Path.cwd() / 'tecton_colab'
ROOT.mkdir(parents=True, exist_ok=True)
""" + f"payload = base64.b64decode({payload!r})\nassert hashlib.sha256(payload).hexdigest() == {digest!r}\n" + """# Reemplazar el código de una versión anterior; runs/, state/ y data/ se conservan.
for folder in ['src', 'configs', 'docs']:
    shutil.rmtree(ROOT / folder, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for item in z.infolist():
        p = Path(item.filename)
        assert not p.is_absolute() and '..' not in p.parts
    z.extractall(ROOT)
print('Proyecto:', ROOT)
""")
    add("code", """import importlib.metadata as metadata
import importlib.util
import subprocess

# Usar las bibliotecas que ya trae Colab; instalar solo lo que falte. Sin reinicios de sesión.
pins = {}
for line in (ROOT / 'requirements-colab.txt').read_text().splitlines():
    if '==' in line and not line.startswith('#'):
        name, version = line.split(';')[0].strip().split('==')
        pins[name.lower()] = version
needed = {'numpy': 'numpy', 'pandas': 'pandas', 'scikit-learn': 'sklearn', 'joblib': 'joblib', 'threadpoolctl': 'threadpoolctl', 'catboost': 'catboost', 'lightgbm': 'lightgbm', 'gdown': 'gdown'}
missing = [name for name, module in needed.items() if importlib.util.find_spec(module) is None]
if missing:
    spec = [f'{n}=={pins[n]}' if n in pins else n for n in missing]
    print('Instalando:', spec)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *spec])
for module in [m for m in sys.modules if m == 'tecton' or m.startswith('tecton.')]:
    del sys.modules[module]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
print('Versiones en uso:', {n: metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm']})
""")
    add("markdown", "## Datos\nDescarga directa de `hackathon_datos.zip` desde `URL_DATOS`; los CSV se copian sin modificar a `data/raw`. Solo si la descarga falla se pide subir el ZIP. No se imprimen filas.\n")
    add("code", """import re
import urllib.request

NAMES = ['entrenamiento.csv', 'prueba_equipos.csv', 'prueba_oculta.csv']
raw = ROOT / 'data/raw'
raw.mkdir(parents=True, exist_ok=True)
DATA_ZIP = ROOT / 'hackathon_datos.zip'


def descargar(url, destino):
    match = re.search(r'/d/([A-Za-z0-9_-]+)', url) or re.search(r'id=([A-Za-z0-9_-]+)', url)
    direct = 'https://drive.google.com/uc?export=download&id=' + match.group(1) if match else url
    try:
        import gdown
        gdown.download(direct, str(destino), quiet=True)
    except Exception as error:
        print('gdown falló, intento descarga directa:', type(error).__name__)
    if not destino.exists() or not zipfile.is_zipfile(destino):
        urllib.request.urlretrieve(direct, destino)
    if not zipfile.is_zipfile(destino):
        destino.unlink(missing_ok=True)
        raise ValueError('La descarga no devolvió un ZIP; revisar permisos del enlace.')


if any(not (raw / name).exists() for name in NAMES):
    if not DATA_ZIP.exists() and URL_DATOS:
        try:
            descargar(URL_DATOS, DATA_ZIP)
        except Exception as error:
            print('No se pudo descargar:', error)
    if not DATA_ZIP.exists() and Path('hackathon_datos.zip').exists():
        shutil.copy2('hackathon_datos.zip', DATA_ZIP)
    if not DATA_ZIP.exists() and IN_COLAB:
        print('Subir hackathon_datos.zip')
        uploaded = files.upload()
        name = next((n for n in uploaded if n.endswith('.zip')), None)
        if name is None:
            raise ValueError('Se esperaba hackathon_datos.zip.')
        DATA_ZIP.write_bytes(uploaded[name])
    if not DATA_ZIP.exists():
        raise FileNotFoundError('Falta hackathon_datos.zip: definir URL_DATOS o dejar el ZIP junto al notebook.')
    with zipfile.ZipFile(DATA_ZIP) as z:
        for item in z.infolist():
            if Path(item.filename).name in NAMES + ['diccionario_datos.csv', 'formato_entrega.csv'] and not item.is_dir():
                (raw / Path(item.filename).name).write_bytes(z.read(item))
faltan = [name for name in NAMES if not (raw / name).exists()]
if faltan:
    raise FileNotFoundError(f'El ZIP no trae {faltan}.')
print('CSV oficiales listos en', raw)
""")
    add("markdown", "## Configuración\nAuditoría estricta del contrato. Solo resúmenes agregados.\n")
    add("code", """from tecton.pipeline import dump_json
from tecton.schema import load_inputs

EMBEDDED_CONFIG = json.loads(""" + repr(json.dumps(config)) + """)
config = json.loads((ROOT / 'configs' / f'{CONFIG_NAME}.json').read_text()) if CONFIG_NAME else EMBEDDED_CONFIG
config = {**config, 'data_dir': 'data/raw', 'strict': True}
if THREADS:
    config = {**config, 'threads': int(THREADS)}
_, audit, _, _ = load_inputs(ROOT / config['data_dir'], True)
dump_json(ROOT / 'configs/colab_run.json', config)
print(json.dumps({'config': config['name'], 'threads': config['threads'], 'audit': audit}, indent=2, ensure_ascii=False))
""")
    add("markdown", "## Entrenamiento\nImprime AUC, RMSE-log, cobertura y segundos por fold. Cada run queda en una carpeta nueva; un fallo no reemplaza el champion.\n")
    add("code", "from tecton.pipeline import run\n\nrun_id = run(ROOT, ROOT / 'configs/colab_run.json')\nprint('Experimento:', run_id)\n")
    add("code", """import pandas as pd
from IPython.display import HTML, display
from tecton.pipeline import compare_runs

pointer = ROOT / 'state' / 'champion.json'
print('Champion:', json.loads(pointer.read_text())['run_id'] if pointer.exists() else 'sin champion')
columns = ['run_id', 'puntos_75_mean', 'auc_mean', 'rmse_log_mean', 'winkler_log_mean', 'coverage_80_mean', 'last_fold_auc', 'stress_auc', 'folds', 'elapsed_seconds']
table = pd.DataFrame(compare_runs(ROOT))
display(table[[c for c in columns if c in table.columns]])
""")
    add("markdown", "## Promoción\nEscribir el `run_id` y ejecutar. El harness rechaza runs incompletos, sin cinco folds y stress, o que no superan al champion bajo el mismo protocolo.\n")
    add("code", """PROMOTE_RUN_ID = None
PROMOTE_REASON = 'Cinco folds, stress de 12 meses y QA revisados'
if PROMOTE_RUN_ID:
    from tecton.pipeline import promote
    print(json.dumps(promote(ROOT, PROMOTE_RUN_ID, PROMOTE_REASON), indent=2, ensure_ascii=False))
""")
    add("code", "display(HTML((ROOT / 'runs' / run_id / 'dashboard.html').read_text()))\n")
    add("markdown", "## Descargas\n1. **Resumen para revisión**: métricas, manifiestos, auditoría y comparación; sin filas, OOF ni predicciones.\n2. **Entrega**: `predicciones.csv` para el formulario, replicabilidad y declaraciones.\n3. **Run para el geovisor**: el run sin `oof.csv`, para `tecton import-run` en la PC.\n")
    add("code", """from tecton.artifacts import review_pack

review = review_pack(ROOT)
print('Resumen para revisión:', review)
if IN_COLAB:
    files.download(str(review))
""")
    add("code", """EXPORT_RUN_ID = None  # None = champion si existe; si no, el último run de esta sesión.
from tecton.artifacts import bundle, run_pack, zip_folder

selected = EXPORT_RUN_ID or (json.loads(pointer.read_text())['run_id'] if pointer.exists() else globals().get('run_id'))
if not selected:
    raise ValueError('Indicar EXPORT_RUN_ID.')
delivery = ROOT / 'delivery' / selected
if not delivery.exists():
    bundle(ROOT, run_id=selected)
csv_entrega = ROOT / 'export' / f'predicciones-{selected}.csv'
csv_entrega.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(delivery / 'predicciones.csv', csv_entrega)
delivery_zip = zip_folder(delivery, ROOT / 'export' / f'entrega-{selected}.zip')
run_zip = run_pack(ROOT, selected)
print('CSV para el formulario:', csv_entrega)
print('Entrega completa:', delivery_zip)
print('Run para geovisor:', run_zip)
if IN_COLAB:
    files.download(str(csv_entrega))
    files.download(str(delivery_zip))
    files.download(str(run_zip))
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "colab": {"name": "tecton_colab.ipynb"}}, "cells": cells}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def bundle(root: Path, run_id=None, demo=False):
    root = Path(root).resolve()
    if run_id is None:
        pointer = root / "state" / ("champion-demo.json" if demo else "champion.json")
        if not pointer.exists():
            raise ValueError("No hay champion. Promover uno o exportar provisionalmente con --run-id.")
        run_id = json.loads(pointer.read_text())["run_id"]
    out = resolve_run(root, run_id)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest.get("qa_passed"):
        raise ValueError("No exportar un experimento incompleto.")
    if manifest["synthetic"] and not demo:
        raise ValueError("Run sintético: añadir --demo para identificar esta entrega como ensayo.")
    verify_run_artifacts(out, manifest)
    frames, _, hashes, _ = load_inputs(root / manifest["config"]["data_dir"], manifest["config"]["strict"])
    if hashes != manifest["data_hashes"]:
        raise ValueError("Los CSV cambiaron desde entrenamiento. Conservar o restaurar los originales.")
    expected = pd.concat([f[KEYS] for f in frames[1:]], ignore_index=True)
    pred = pd.read_csv(out / "predicciones.csv", dtype={"DIVIPOLA": "string"})
    validate_predictions(pred, expected)
    target = root / "delivery" / run_id
    target.mkdir(parents=True, exist_ok=False)
    for name in ["predicciones.csv", "replicabilidad.csv", "dashboard.html", "metrics.json", "manifest.json", "source.zip"]:
        shutil.copy2(out / name, target / name)
    with zipfile.ZipFile(out / "source.zip") as source:
        (target / "requirements-colab.txt").write_bytes(source.read("requirements-colab.txt"))
    portable_notebook(out / "source.zip", manifest["config"], target / "tecton_colab.ipynb")
    ai_file = root / "docs" / "ai_usage.csv"
    if ai_file.exists():
        shutil.copy2(ai_file, target / "ai_usage.csv")
    else:
        (target / "ai_usage.csv").write_text("herramienta,version,uso,revision_humana\nClaude Code,COMPLETAR,Codigo y experimentos,COMPLETAR\nChatGPT Codex,COMPLETAR,Scaffold e instrucciones,COMPLETAR\n", encoding="utf-8")
    sources_file = root / "docs" / "fuentes_datos.csv"
    if sources_file.exists():
        shutil.copy2(sources_file, target / "fuentes_datos.csv")
    if (root / "dashboard/dist/index.html").exists():
        from tecton.dashboard import export_dashboard

        geo = root / "data/geography/municipios.geojson"
        export_dashboard(root, run_id, geo if geo.exists() and not manifest["synthetic"] else None, target / "geovisor", demo)
    (target / "LEEME.txt").write_text("Entrega reproducible. No incluye CSV oficiales. Ejecutar tecton_colab.ipynb y aportar los tres CSV. Completar ai_usage.csv. La replicabilidad contiene límites pendientes, no equivalencias garantizadas. No publicar datos o predicciones del reto. Si manifest.synthetic=true, es solo un ensayo y no una entrega oficial.\n", encoding="utf-8")
    return target


def zip_folder(folder: Path, target: Path, exclude=()):
    folder, target = Path(folder), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name not in exclude:
                archive.write(path, Path(folder.name) / path.relative_to(folder))
    temporary.replace(target)
    return target


def review_pack(root: Path):
    """ZIP de agregados para revisar fuera de Colab; excluye filas, OOF, predicciones y modelos."""
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / "export" / f"resumen-{stamp}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in (root / "runs").iterdir() if p.is_dir()) if (root / "runs").is_dir() else []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for run_dir in runs:
            for name in REVIEW_FILES:
                if (run_dir / name).is_file():
                    archive.write(run_dir / name, f"resumen/runs/{run_dir.name}/{name}")
        for name in ["champion.json", "champion-demo.json", "promotion_history.jsonl", "FROZEN.json", "submissions.jsonl"]:
            if (root / "state" / name).is_file():
                archive.write(root / "state" / name, f"resumen/state/{name}")
        archive.writestr("resumen/compare.json", json.dumps(compare_runs(root), indent=2, ensure_ascii=False))
    return target


def run_pack(root: Path, run_id: str):
    """Run completo para importar en la PC; sin oof.csv porque contiene targets."""
    out = resolve_run(Path(root), run_id)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest.get("qa_passed"):
        raise ValueError("Solo se exportan runs completos con QA.")
    verify_run_artifacts(out, manifest)
    return zip_folder(out, Path(root) / "export" / f"run-{run_id}.zip", exclude={"oof.csv"})


def import_run(root: Path, archive_path: Path):
    """Importa a runs/ un run entrenado en Colab, verificando hashes antes de aceptarlo."""
    root = Path(root).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = [Path(n) for n in archive.namelist() if not n.endswith("/")]
        if any(p.is_absolute() or ".." in p.parts for p in names):
            raise ValueError("ZIP con rutas inseguras.")
        tops = {p.parts[0] for p in names}
        if len(tops) != 1:
            raise ValueError("El ZIP debe contener una sola carpeta de run.")
        run_id = tops.pop()
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
            raise ValueError("run_id inválido en el ZIP.")
        if any(p.name == "oof.csv" for p in names):
            raise ValueError("El ZIP incluye oof.csv con targets; exportarlo con run_pack.")
        target = root / "runs" / run_id
        if target.exists():
            raise ValueError(f"El run ya existe localmente: {run_id}.")
        (root / "runs").mkdir(exist_ok=True)
        staging = root / "runs" / f".import-{run_id}"
        shutil.rmtree(staging, ignore_errors=True)
        archive.extractall(staging)
    try:
        staged = staging / run_id
        manifest = json.loads((staged / "manifest.json").read_text())
        if manifest.get("run_id") != run_id or manifest["status"] != "complete" or not manifest.get("qa_passed"):
            raise ValueError("Manifest incompleto o no corresponde al run.")
        verify_run_artifacts(staged, manifest)
        if sha256(staged / "source.zip") != manifest["source_sha256"]:
            raise ValueError("source.zip no coincide con el manifest.")
        staged.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return run_id
