"""Valida el notebook de Colab y, opcionalmente, lo ejecuta con los datos oficiales locales."""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
parser.add_argument("--execute", action="store_true", help="Ejecutar todas las celdas en este intérprete, sin kernel Jupyter")
parser.add_argument("--workdir", type=Path, help="Carpeta de trabajo vacía (simula /content de Colab)")
parser.add_argument("--data-zip", type=Path, help="hackathon_datos.zip local para no descargarlo")
parser.add_argument("--config-name", help="Sustituir CONFIG_NAME para una prueba más corta")
args = parser.parse_args()
notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4
cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
for i, source in enumerate(cells):
    compile(source, f"{args.notebook}:cell-{i}", "exec")
if args.execute:
    if args.config_name:
        cells = [c.replace("CONFIG_NAME = None", f"CONFIG_NAME = {args.config_name!r}") for c in cells]
    workdir = (args.workdir or Path.cwd()).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if args.data_zip:
        (workdir / "hackathon_datos.zip").write_bytes(args.data_zip.read_bytes())
    os.chdir(workdir)
    namespace = {"__name__": "__main__", "display": print}
    for i, source in enumerate(cells):
        exec(compile(source, f"{args.notebook}:cell-{i}", "exec"), namespace)
    print("Celdas ejecutadas en", workdir)
print("Notebook válido:", args.notebook)
