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


def portable_notebook(source_zip: Path, config: dict, target: Path):
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
Código incluido y verificado por SHA256; el notebook llama al mismo motor Python, no reimplementa el entrenamiento. CPU, Python 3.11–3.13.
No contiene targets ni datos oficiales: los tres CSV se suben en la celda de datos.

Reglas del reto: no compartir los datos fuera del equipo, no editar predicciones a mano, cierre 15:00 hora Colombia por el formulario oficial.

Flujo: ejecutar todo una vez. Para otro experimento, cambiar `CONFIG_NAME` en la primera celda y ejecutar de nuevo desde **Configuración**; los datos ya cargados se reutilizan.
""")
    add("code", """from pathlib import Path

INSTALL_DEPENDENCIES = True
USE_SYNTHETIC_DATA = False
# None = configuración embebida (la del run de entrega). Otras: 'first_submission', 'baseline', 'catboost'.
CONFIG_NAME = None
# Hilos de CPU; None conserva los de la configuración. Cambiarlos no cambia el protocolo temporal.
THREADS = None
# True guarda runs/ y state/ en el Drive del equipo para sobrevivir desconexiones. No compartir esa carpeta.
USE_DRIVE = False
DRIVE_FOLDER = 'TECTON_PNUD'
# Enlace de Drive a hackathon_datos.zip compartido por el PNUD. Pegarlo solo en la copia de Colab del equipo; no versionarlo.
URL_DATOS = ''
# Alternativa: subir hackathon_datos.zip con el ícono de carpeta, o los tres CSV cuando se soliciten.
DATA_ZIP = Path('hackathon_datos.zip')
LOCAL_DATA_DIR = Path('data/raw')
""")
    add("code", """try:
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
print('Proyecto:', ROOT)
""")
    add("code", "import base64, hashlib, io, json, shutil, zipfile\n" + f"payload = base64.b64decode({payload!r})\nassert hashlib.sha256(payload).hexdigest() == {digest!r}\n" + """# Reemplazar el código de una versión anterior; runs/, state/ y data/ se conservan.
for folder in ['src', 'configs', 'docs']:
    shutil.rmtree(ROOT / folder, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for item in z.infolist():
        p = Path(item.filename)
        assert not p.is_absolute() and '..' not in p.parts
    z.extractall(ROOT)
""")
    add("code", """import importlib.metadata as metadata
import subprocess
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise RuntimeError('Este scaffold requiere Python 3.11–3.13; seleccionar un runtime compatible.')
if INSTALL_DEPENDENCIES:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'requirements-colab.txt')])
pins = {}
for line in (ROOT / 'requirements-colab.txt').read_text().splitlines():
    if '==' in line and not line.startswith('#'):
        name, version = line.split(';')[0].strip().split('==')
        pins[name.lower()] = version
stale = []
for name, module in [('numpy', 'numpy'), ('pandas', 'pandas'), ('scikit-learn', 'sklearn'), ('catboost', 'catboost')]:
    loaded = sys.modules.get(module)
    found = getattr(loaded, '__version__', None) if loaded else metadata.version(name)
    if found != pins[name]:
        stale.append(f'{name} {found} != {pins[name]}')
if stale:
    raise RuntimeError('Versiones distintas del lock: ' + ', '.join(stale) + '. Reiniciar la sesión y ejecutar desde el inicio.')
# Descargar módulos tecton de una versión anterior del código en esta misma sesión.
for module in [m for m in sys.modules if m == 'tecton' or m.startswith('tecton.')]:
    del sys.modules[module]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
print('Dependencias del lock:', {n: pins[n] for n in ['numpy', 'pandas', 'scikit-learn', 'catboost']})
""")
    add("markdown", "## Datos\nIgual que el notebook base del PNUD: pegar `URL_DATOS` o subir `hackathon_datos.zip`. Si no hay ZIP, se piden los tres CSV con su nombre exacto. Se cargan una vez por sesión (o una sola vez si `USE_DRIVE=True`). No se imprimen filas. Para un ensayo, activar `USE_SYNTHETIC_DATA` en la primera celda.\n")
    add("code", """from tecton.demo import generate

NAMES = ['entrenamiento.csv', 'prueba_equipos.csv', 'prueba_oculta.csv']
if USE_SYNTHETIC_DATA:
    generate(ROOT / 'data/synthetic')
    print('Datos INVENTADOS en data/synthetic; solo ensayo.')
else:
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    missing = [name for name in NAMES if not (raw / name).exists()]
    if missing and URL_DATOS and not DATA_ZIP.exists():
        import re
        import gdown
        # Usar el id del archivo: compatible con versiones de gdown con y sin el argumento fuzzy.
        file_id = re.search(r'/d/([A-Za-z0-9_-]+)', URL_DATOS) or re.search(r'id=([A-Za-z0-9_-]+)', URL_DATOS)
        gdown.download('https://drive.google.com/uc?id=' + (file_id.group(1) if file_id else URL_DATOS), str(DATA_ZIP), quiet=True)
    if missing and DATA_ZIP.exists():
        # El ZIP oficial puede traer subcarpetas; se copian los CSV por nombre exacto, sin modificarlos.
        with zipfile.ZipFile(DATA_ZIP) as z:
            for item in z.infolist():
                if Path(item.filename).name in NAMES + ['diccionario_datos.csv'] and not item.is_dir():
                    (raw / Path(item.filename).name).write_bytes(z.read(item))
        missing = [name for name in NAMES if not (raw / name).exists()]
    if missing and IN_COLAB:
        uploaded = files.upload()
        for name in missing:
            if name not in uploaded:
                raise ValueError('Falta el archivo con nombre exacto: ' + name)
            (raw / name).write_bytes(uploaded[name])
    else:
        for name in missing:
            shutil.copy2(LOCAL_DATA_DIR / name, raw / name)
    print('CSV oficiales listos en', raw)
""")
    add("markdown", "## Configuración y auditoría\nSolo resúmenes agregados: filas, municipios, fechas, nulos y prevalencia.\n")
    add("code", """from tecton.pipeline import dump_json
from tecton.schema import load_inputs

EMBEDDED_CONFIG = json.loads(""" + repr(json.dumps(config)) + """)
SMOKE = {None: 'smoke', 'first_submission': 'smoke', 'baseline': 'baseline-smoke', 'catboost': 'catboost-smoke'}
if USE_SYNTHETIC_DATA:
    config = json.loads((ROOT / 'configs' / f"{SMOKE.get(CONFIG_NAME, CONFIG_NAME)}.json").read_text())
elif CONFIG_NAME:
    config = json.loads((ROOT / 'configs' / f'{CONFIG_NAME}.json').read_text())
else:
    config = EMBEDDED_CONFIG
if config['data_dir'] != ('data/synthetic' if USE_SYNTHETIC_DATA else 'data/raw'):
    raise ValueError('La configuración no corresponde al modo de datos elegido.')
if THREADS:
    config = {**config, 'threads': int(THREADS)}
_, audit, _, synthetic = load_inputs(ROOT / config['data_dir'], config['strict'])
dump_json(ROOT / 'configs/colab_run.json', config)
print(json.dumps({'config': config['name'], 'threads': config['threads'], 'synthetic': synthetic, 'audit': audit}, indent=2, ensure_ascii=False))
""")
    add("markdown", "## Entrenamiento\nImprime AUC, RMSE-log, cobertura y segundos por fold. Cada run queda en una carpeta nueva; un fallo no reemplaza el champion.\n")
    add("code", "from tecton.pipeline import run\n\nrun_id = run(ROOT, ROOT / 'configs/colab_run.json')\nprint('Experimento:', run_id)\n")
    add("code", """import pandas as pd
from IPython.display import HTML, display
from tecton.pipeline import compare_runs

pointer = ROOT / 'state' / ('champion-demo.json' if USE_SYNTHETIC_DATA else 'champion.json')
print('Champion:', json.loads(pointer.read_text())['run_id'] if pointer.exists() else 'sin champion')
display(pd.DataFrame(compare_runs(ROOT)))
""")
    add("markdown", "## Promoción\nEscribir el `run_id` y ejecutar. El harness rechaza runs incompletos, sin cinco folds/stress o que no superan al champion bajo el mismo protocolo.\n")
    add("code", """PROMOTE_RUN_ID = None
PROMOTE_REASON = 'Cinco folds, stress de 12 meses y QA revisados'
if PROMOTE_RUN_ID:
    from tecton.pipeline import promote
    print(json.dumps(promote(ROOT, PROMOTE_RUN_ID, PROMOTE_REASON, demo=USE_SYNTHETIC_DATA), indent=2, ensure_ascii=False))
""")
    add("code", "display(HTML((ROOT / 'runs' / run_id / 'dashboard.html').read_text()))\n")
    add("markdown", "## Descargas\n1. **Resumen para revisión**: métricas, manifiestos, auditoría y comparación. Sin filas, sin OOF ni predicciones; es lo que se comparte con el asistente de código.\n2. **Entrega**: CSV, notebook reproducible, replicabilidad y declaraciones. Va al formulario oficial.\n3. **Run para el geovisor**: el run sin `oof.csv`, para importarlo en la PC con `tecton import-run`.\n")
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
    bundle(ROOT, run_id=selected, demo=USE_SYNTHETIC_DATA)
delivery_zip = zip_folder(delivery, ROOT / 'export' / f'entrega-{selected}.zip')
run_zip = run_pack(ROOT, selected)
print('Entrega:', delivery_zip)
print('Run para geovisor:', run_zip)
print('Completar ai_usage.csv y revisar replicabilidad antes de enviar. El formulario lo envía el equipo.')
if IN_COLAB:
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
