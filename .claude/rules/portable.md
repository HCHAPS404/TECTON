---
paths:
  - "src/**/*.py"
  - "notebooks/*.ipynb"
  - "scripts/*.py"
  - "pyproject.toml"
---
# Portabilidad

- Usar pathlib y rutas relativas a la raíz. Evitar /home/usuario, /content/drive y cwd implícito en bibliotecas internas.
- CPU por defecto, sin CUDA ni instalación de paquetes Arch dentro del notebook.
- requirements-colab.txt es exportado de uv.lock, no mantenido a mano.
- Generar notebook a partir de source.zip del run seleccionado. No copiar celdas de lógica ML.
- Colab instala dependencias antes de importar bibliotecas; recomendar runtime limpio/reinicio si ya hay versiones cargadas.
- No empaquetar data/raw, OOF, credenciales ni entorno .venv en el scaffold.
