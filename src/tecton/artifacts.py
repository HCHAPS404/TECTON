import base64
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from tecton.pipeline import resolve_run, verify_run_artifacts
from tecton.schema import KEYS, load_inputs, validate_predictions


def snapshot(root: Path, target: Path):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        paths = list((root / "src").rglob("*.py")) + list((root / "configs").glob("*.json"))
        paths += [root / name for name in ["pyproject.toml", "uv.lock", "requirements-colab.txt"]]
        for path in sorted(paths):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(root))


def feature_registry(features):
    rows = []
    for name in features:
        source, proxy, limit = "Derivada", "Calendario local equivalente", "Reproducir la misma transformación."
        if name in ["DIVIPOLA", "departamento"]:
            source, proxy, limit = "DANE / derivada de DIVIPOLA", "geoBoundaries / código administrativo local", "Los niveles administrativos no son idénticos entre países."
        elif "densidad" in name:
            source, proxy, limit = "TerriData DNP", "WorldPop: población agregada al área administrativa", "Alinear año, unidad y límites."
        elif "ingresos" in name or "inversion" in name:
            source, proxy, limit = "TerriData DNP", "Finanzas públicas subnacionales abiertas del país", "No existe aquí un sustituto municipal global verificado; equivalencia pendiente."
        elif name == "nbi_pct":
            source, proxy, limit = "TerriData / Censo DANE 2018", "Global MPI o encuestas nacionales abiertas", "Proxy de privación: no equivale exactamente a NBI; revisar resolución y cobertura."
        elif any(word in name for word in ["elevacion", "pendiente"]):
            source, proxy, limit = "Copernicus DEM + agregación municipal", "Copernicus DEM global", "Para oni_pendiente usar también NOAA ONI; mantener resolución/unidades."
        elif name in ["ONI", "fase_enso"]:
            source, proxy, limit = "NOAA CPC", "Misma serie ONI de NOAA", "Señal compartida nacional; no mide lluvia local."
        elif name.startswith("history_"):
            source, proxy, limit = "Targets del entrenamiento, con corte temporal", "Registro abierto de desastres local con unidad administrativa y mes", "No se verificó una equivalencia global municipal-mensual; documentar disponibilidad o eliminar feature al transferir."
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

    add("markdown", "# TECTON PNUD — notebook portátil\nCódigo incluido y verificado por SHA256. Los datos oficiales se aportan por separado. CPU, Python 3.11–3.13.\nNo contiene targets ni datos oficiales. Ejecutar desde el inicio en un runtime limpio.\nEste notebook llama al mismo motor de Python local, no reimplementa el entrenamiento.\n")
    add("code", "from pathlib import Path\nINSTALL_DEPENDENCIES = True\nUSE_SYNTHETIC_DATA = False\nLOCAL_DATA_DIR = Path('data/raw')\nROOT = Path.cwd() / 'tecton_colab'\nROOT.mkdir(exist_ok=True)\n")
    add("code", "import base64, hashlib, io, json, zipfile\n" + f"payload = base64.b64decode({payload!r})\nassert hashlib.sha256(payload).hexdigest() == {digest!r}\n" + "with zipfile.ZipFile(io.BytesIO(payload)) as z:\n    for item in z.infolist():\n        p = Path(item.filename)\n        assert not p.is_absolute() and '..' not in p.parts\n    z.extractall(ROOT)\n")
    add("code", "import sys, subprocess\nif not ((3, 11) <= sys.version_info[:2] < (3, 14)):\n    raise RuntimeError('Este scaffold requiere Python 3.11–3.13; seleccionar un runtime compatible.')\nif INSTALL_DEPENDENCIES:\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'requirements-colab.txt')])\nsys.path.insert(0, str(ROOT / 'src'))\n")
    add("markdown", "## Datos\nEn Colab se solicitarán los tres CSV oficiales. En local ajustar LOCAL_DATA_DIR. Para un ensayo, activar USE_SYNTHETIC_DATA en la primera celda.\nLa instalación debe ejecutarse antes de importar numpy/pandas; si ya estaban cargados con otras versiones, reiniciar la sesión.\n")
    add("code", "import shutil\nfrom tecton.demo import generate\nfrom tecton.pipeline import dump_json\nconfig = json.loads(" + repr(json.dumps(config)) + ")\nif USE_SYNTHETIC_DATA:\n    config = json.loads((ROOT / 'configs/smoke.json').read_text())\n    generate(ROOT / config['data_dir'])\nelse:\n    raw = ROOT / 'data/raw'\n    raw.mkdir(parents=True, exist_ok=True)\n    try:\n        from google.colab import files\n    except ImportError:\n        for name in ['entrenamiento.csv', 'prueba_equipos.csv', 'prueba_oculta.csv']:\n            shutil.copy2(LOCAL_DATA_DIR / name, raw / name)\n    else:\n        uploaded = files.upload()\n        for name in ['entrenamiento.csv', 'prueba_equipos.csv', 'prueba_oculta.csv']:\n            if name not in uploaded:\n                raise ValueError('Falta el archivo con nombre exacto: ' + name)\n            (raw / name).write_bytes(uploaded[name])\ndump_json(ROOT / 'configs/colab_run.json', config)\n")
    add("code", "from tecton.pipeline import run\nrun_id = run(ROOT, ROOT / 'configs/colab_run.json')\nprint('Experimento:', run_id)\n")
    add("code", "from IPython.display import HTML, display\ndisplay(HTML((ROOT / 'runs' / run_id / 'dashboard.html').read_text()))\n")
    add("code", "from tecton.artifacts import bundle\nbundle_dir = bundle(ROOT, run_id=run_id, demo=USE_SYNTHETIC_DATA)\nprint('Entrega local:', bundle_dir)\nprint('Completar ai_usage.csv y revisar tabla de replicabilidad antes de enviar.')\n# En Colab, para descargar el CSV: files.download(str(bundle_dir / 'predicciones.csv'))\n")
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
    if (root / "dashboard/dist/index.html").exists():
        from tecton.dashboard import export_dashboard

        geo = root / "data/geography/municipios.geojson"
        export_dashboard(root, run_id, geo if geo.exists() and not manifest["synthetic"] else None, target / "geovisor", demo)
    (target / "LEEME.txt").write_text("Entrega reproducible. No incluye CSV oficiales. Ejecutar tecton_colab.ipynb y aportar los tres CSV. Completar ai_usage.csv. La replicabilidad contiene límites pendientes, no equivalencias garantizadas. No publicar datos o predicciones del reto. Si manifest.synthetic=true, es solo un ensayo y no una entrega oficial.\n", encoding="utf-8")
    return target
