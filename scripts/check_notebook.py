"""Validación estructural local. Ejecución real mediante nbclient opcional."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
parser.add_argument("--execute-synthetic", action="store_true")
parser.add_argument("--execute-python-synthetic", action="store_true", help="Ejecutar celdas Python sin sockets de Jupyter; no equivale a probar un kernel Colab")
args = parser.parse_args()
notebook = nbformat.read(args.notebook, as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        compile(cell.source, str(args.notebook), "exec")
if args.execute_synthetic or args.execute_python_synthetic:
    for cell in notebook.cells:
        if cell.cell_type == "code" and "INSTALL_DEPENDENCIES = True" in cell.source:
            cell.source = cell.source.replace("INSTALL_DEPENDENCIES = True", "INSTALL_DEPENDENCIES = False").replace("USE_SYNTHETIC_DATA = False", "USE_SYNTHETIC_DATA = True")
    if args.execute_python_synthetic:
        namespace = {"__name__": "__main__"}
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, f"{args.notebook}:cell-{i}", "exec"), namespace)
        print("Celdas Python ejecutadas sin kernel Jupyter, con datos inventados.")
    else:
        NotebookClient(notebook, timeout=180, kernel_name="python3").execute()
        target = args.notebook.with_name(args.notebook.stem + "_executed.ipynb")
        nbformat.write(notebook, target)
        print("Notebook ejecutado con datos inventados:", target)
print("Notebook válido:", args.notebook)
