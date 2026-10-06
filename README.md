# TECTON PNUD · Python local + Claude Code + Colab

Proyecto base ejecutable para trabajar en Arch Linux, conservar resultados verificables y entregar un notebook Colab con el mismo código. No contiene datos oficiales ni resultados competitivos. Preparación para el martes 6 de octubre de 2026; modelado 10:00–14:00 hora Bogotá.

La división de trabajo está en [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md). Claude lleva datos y modelo. Cursor implementa la estética de gráficas y mapas. Laura pule y mejora la interfaz. En este repositorio la raíz ya es el proyecto.

## 1. Instalación en tu PC

Arch puede actualizar su Python del sistema a una versión distinta de Colab. Este proyecto usa Python 3.12 administrado por uv y un entorno `.venv` separado. No instalar sus bibliotecas con sudo pip.

```bash
# En Arch ya instalado; usar la base de paquetes actual, sin ejecutar pacman -Sy aislado.
sudo pacman -S --needed git uv unzip curl

# Este repositorio ya es la raíz del proyecto.
cd TECTON

uv python install 3.12
uv sync --frozen
uv run --frozen python -m tecton doctor
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check src tests scripts
uv run --frozen python -m tecton synthetic
uv run --frozen python -m tecton run --config configs/smoke.json
uv run --frozen python -m tecton run --config configs/catboost-smoke.json
uv run --frozen python scripts/check_notebook.py notebooks/tecton_colab.ipynb --execute-synthetic

# Crear historial local. Usar tu identidad git ya configurada.
git init
git add .
git commit -m "Base TECTON PNUD verificada"
```

El ZIP no contiene `.venv` ni datos oficiales; `examples/` contiene un ensayo identificado. Necesitas internet para descargar Python y las dependencias una vez; después se reutilizan. No cambiar versiones en mitad del evento. Si falta una biblioteca, reinstalar desde el lock, no actualizar todo el stack.

Claude Code CLI, instalador nativo oficial para Linux:

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version
claude
```

Completar el inicio de sesión desde tu PC. El instalador nativo no necesita una cadena Node/npm para este proyecto. Elegir un modelo habilitado en tu cuenta y registrar el nombre real junto con `claude --version` en `docs/ai_usage.csv`. No hace falta seleccionar un modelo específico para que el scaffold funcione.

## 2. Qué aporta cada capa

| Capa | Archivo o componente | Función |
|---|---|---|
| Prompt inicial | docs/PROMPT_INICIAL.md | Objetivo y primera ejecución |
| Memoria de proyecto | CLAUDE.md | Arquitectura, comandos e invariantes breves |
| Rules | .claude/rules/ | Reglas de temporalidad, experimentos y portabilidad por rutas |
| Skills | .claude/skills/ | Procedimientos bootstrap, audit, experiment y deliver |
| Hooks | .claude/hooks/ + settings.json | Protección de archivos mediante herramientas de lectura/edición y contexto de sesión |
| Harness de ML | python -m tecton | Auditoría, entrenamiento, comparación, promoción, entrega y reloj |
| Scaffold | src/ + configs/ + tests/ + notebooks/ | Estructura ejecutable que materializa todo lo anterior |
| Historial | git local + runs/ + state/ | Cambios de código y evidencia de cada experimento |

Las instrucciones guían a Claude; el harness comprueba condiciones mediante código. Los hooks de archivo no aíslan Bash ni otros procesos: no son una garantía de confidencialidad. El flujo evita llevar filas al contexto; el equipo debe respetar los T&C y los permisos del proveedor de IA. No usar bypass de permisos ni añadir MCP/Agent SDK para esta jornada.

En Claude, pegar `docs/PROMPT_INICIAL.md` y después usar cuando convenga:

```text
/tecton-bootstrap
/tecton-audit
/tecton-experiment Probar una configuración CatBoost con el mismo protocolo del baseline
/tecton-deliver champion
```

`tecton-deliver` requiere invocación expresa y solo prepara archivos locales; no envía el formulario. Revisar carga de instrucciones con `/context` y, si hace falta, `/memory`. No usar `/init` para sustituir archivos ya preparados.

## 3. Estructura

| Ruta | Contenido |
|---|---|
| src/tecton/schema.py | Lectura, auditoría, hashes y contrato exacto de CSV |
| src/tecton/features.py | Covariables + historia causal/estacional suavizada |
| src/tecton/models.py | HistGradientBoosting de respaldo y CatBoost principal; cuatro modelos por motor |
| src/tecton/metrics.py | AUC, RMSE log1p, Winkler log1p alfa 0.2 y cobertura |
| src/tecton/pipeline.py | Folds, entrenamiento final, OOF, snapshots y promoción |
| src/tecton/artifacts.py | Notebook autónomo, dashboard HTML de respaldo, tabla de replicabilidad y bundle |
| src/tecton/dashboard.py | Exportador del geovisor y descarga de geometrías DANE |
| dashboard/ | React + Kepler.gl, lockfile, tests y versión compilada dist/ en el ZIP |
| examples/datos-ensayo.json | Resultado inventado para probar la interfaz, sin ubicaciones reales |
| configs/first_submission.json | Una validación de 12 meses y modelo rápido para entrega provisional |
| configs/baseline.json | Cinco folds + stress de 12 meses, motor sklearn |
| configs/catboost.json | Mismo protocolo, motor CatBoost |
| configs/smoke.json | Ensayo sintético breve; no producción |
| data/raw/ | Tres CSV oficiales originales, ignorados por git |
| data/synthetic/ | Datos inventados de prueba, ignorados por git |
| runs/RUN_ID/ | Config, auditoría, métricas, OOF, modelo, CSV y snapshot de código |
| state/ | Puntero del champion, historial de promociones, congelación y entregas |
| delivery/RUN_ID/ | Copia revisable para entrega; sin CSV originales |
| notebooks/tecton_colab.ipynb | Plantilla autónoma para ejecutar el mismo motor |
| docs/ | Contrato, estado, prompts, diseño, IA y verificación |

## 4. Flujo de trabajo durante el reto

**Primero una entrega provisional válida.** Copiar los tres archivos oficiales a data/raw, conservando los nombres exactos y sin editar filas.

```bash
uv run --frozen python -m tecton audit --config configs/baseline.json
uv run --frozen python -m tecton run --config configs/first_submission.json
# El comando anterior imprime un RUN_ID; sustituir el texto literal siguiente.
uv run --frozen python -m tecton bundle --run-id RUN_ID
```

Esta primera versión ofrece una validación futura de 12 meses, no una evaluación de cinco folds. Puede exportarse provisionalmente, pero no convertirse en champion real del harness.

**Después una base sólida.** Entrenar los dos motores bajo los mismos folds y la misma semilla.

```bash
uv run --frozen python -m tecton run --config configs/baseline.json
uv run --frozen python -m tecton run --config configs/catboost.json
uv run --frozen python -m tecton compare
uv run --frozen python -m tecton promote RUN_ID --reason "Cinco folds, stress de 12 meses y QA revisados"
```

Primera promoción real: exige run completo, cinco folds, auditoría estricta y stress de 12 meses. Posteriores promociones: mismos datos/protocolo, +0.002 AUC medio, mejora en al menos 60% de folds, pérdida máxima 0.005 en último fold/stress, y límites de degradación de RMSE (3%) y Winkler (10%). Son umbrales operativos ajustables mediante una decisión documentada, no una prueba de significancia ni garantía de mejora oculta. No cambiarlos para hacer pasar un candidato concreto.

Cada run tiene carpeta nueva. El champion apunta a un run anterior conservado; entrenar otra config no lo reemplaza. Sus predicciones y snapshot continúan disponibles incluso si el nuevo experimento falla. Comparar mean/dispersion y folds recientes antes de promover. No confiar en diferencias de 0.0001 ni en resultados sintéticos.

**Mejorar una hipótesis.** Copiar config, cambiar una feature o hiperparámetro, ejecutar y comparar. Conservar folds y semilla. Registrar idea, run_id y decisión en docs/ESTADO.md. El scaffold inicial no implementa LightGBM, XGBoost, hurdle, ensembles ni calibración conformal; añadir solo uno si queda tiempo y demuestra ganancia bajo el protocolo común.

**Cerrar.** Elegir el run, exportar y verificar con el validador oficial del PNUD en cuanto esté disponible. Resolver cualquier diferencia de contrato explícitamente, no desactivar strict.

```bash
uv run --frozen python -m tecton bundle
uv run --frozen python -m tecton freeze
uv run --frozen python -m tecton clock
# Después de enviar manualmente y confirmar recepción válida:
uv run --frozen python -m tecton record-submission RUN_ID --received-at 2026-10-06T13:30:00-05:00
```

Bundle no modifica el champion. No sobrescribe una carpeta delivery existente: si ya se generó, revisar esa copia. Freeze bloquea promociones reales posteriores y conserva el champion; los runs nuevos pueden continuar separados. El reloj calcula 30 minutos desde las horas de recepción registradas por el equipo; no conoce la validez del formulario ni envía nada.

## 5. Python local → Colab

```bash
# Regenerar plantilla si cambia el código/config antes de entrenar:
uv run --frozen python -m tecton notebook --config configs/baseline.json

# Solo si se decidió cambiar dependencias antes del evento:
uv lock
uv export --frozen --no-dev --no-emit-project --no-hashes --format requirements.txt --output-file requirements-colab.txt
```

uv.lock es la fuente de versiones; requirements-colab.txt es un export para pip. El notebook lleva un ZIP de código embebido, comprueba su SHA256, instala requisitos, solicita los tres CSV y llama a `tecton.pipeline.run`. No necesita GitHub, tu ruta de Arch ni un servidor local. El notebook dentro de delivery usa el snapshot del run elegido, no el código que se haya editado después.

Abrir `notebooks/tecton_colab.ipynb` o el notebook de delivery en Colab. Mantener `INSTALL_DEPENDENCIES=True`, `USE_SYNTHETIC_DATA=False` y un runtime limpio con Python 3.11–3.13. Si ya se importaron bibliotecas antes de cambiarlas, reiniciar. Para comprobar sin datos reales, activar USE_SYNTHETIC_DATA. Las versiones cambian en los runtimes de Colab: confirmar su versión real, no asumir igualdad binaria entre Linux local y Colab. También existe runtime local de Colab, pero no es necesario para este flujo.

## 6. Límites y prioridades

- No se conoce todavía la calidad sobre datos oficiales, la duración en tu PC ni las reglas completas. Los parámetros son puntos de partida.
- Prioridad: riesgo (50%), magnitud (20%), intervalo (5%). Los otros entregables siguen siendo obligatorios.
- Cuantiles entrenados en log1p, expm1 al entregar; cruces corregidos por orden. Reportar cobertura empírica sin prometer 80% por municipio o ante cambio temporal.
- No usar como X los conteos/eventos/familias/viviendas del mes objetivo. El historial implementado usa solo tiene_evento y personas_desplazadas anteriores.
- No se incluyen fuentes externas ni llamadas de red en el modelado. ONI y las demás covariables vienen del dataset oficial.
- El dashboard de respaldo es HTML estático con métricas y una tabla filtrable de 40 combinaciones de mayor riesgo; no es un geovisor completo. Ampliarlo a mapa solo después de asegurar una entrega sólida.
- La tabla cubre cada feature usada y marca equivalencias internacionales no verificadas, especialmente finanzas e historial municipal. Revisarla para la rúbrica real.
- Completar ai_usage.csv y contrastar versión/modelo con la interfaz real de ambas herramientas. No subir filas del reto a prompts ni repositorios.
- Medir tiempo/uso de memoria en la PC. Si 500 iteraciones no caben, probar una config menor antes de competir. Empezar con una semilla y cuatro hilos; bajar a dos si el portátil lo necesita.
- En este entorno se verificó el motor sklearn. CatBoost se instaló/importó, pero entrenarlo falla por acceso restringido a /proc/self/statm; el kernel Jupyter también queda bloqueado por conexiones locales. Verificar ambos en la PC/Colab antes de depender de ellos. La prueba de celdas en un proceso Python separado no reemplaza la prueba real en Colab.

## 7. Documentación oficial consultada

- Claude Code: https://code.claude.com/docs/en/overview
- Memoria y rules: https://code.claude.com/docs/en/memory
- Skills: https://code.claude.com/docs/en/skills
- Hooks: https://code.claude.com/docs/en/hooks
- uv y export: https://docs.astral.sh/uv/concepts/projects/export/
- Paquete uv Arch: https://archlinux.org/packages/extra/x86_64/uv/
- Colab runtimes: https://research.google.com/colaboratory/runtime-version-faq.html
- Colab local: https://research.google.com/colaboratory/local-runtimes.html
- CatBoost cuantiles: https://catboost.ai/docs/en/concepts/loss-functions-regression

Instrucciones y enlaces contrastados el 6 de octubre de 2026. Ver docs/VERIFICACION.md para pruebas realmente ejecutadas en este scaffold.

## Geovisor local con Kepler.gl

El ZIP incluye el visor compilado. Desde la raíz:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory dashboard/dist
```

Abrir http://127.0.0.1:8765. Cargar `examples/datos-ensayo.json` para comprobar el flujo con datos inventados. Para el run real, ejecutar `uv run --frozen python -m tecton geography` y `uv run --frozen python -m tecton dashboard --run-id RUN_ID --geojson data/geography/municipios.geojson`. La geometría DANE no está incluida: descarga o carga manual pendiente en la PC. Ver **docs/DASHBOARD.md** y el prompt **docs/PROMPT_DASHBOARD.md**.

React/Kepler presenta resultados; Python mantiene el motor local y Colab. No se publican datos. El HTML sencillo sigue disponible como respaldo. La compilación y los contratos se probaron; la comprobación visual de WebGL queda pendiente en el navegador de la PC.
