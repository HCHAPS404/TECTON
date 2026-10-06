# TECTON · Hackathon PNUD Colombia 2026

Sistema predictivo de riesgo de desplazamiento por desastres climáticos para los 1.102 municipios de Colombia, mes a mes. Desarrollado para la Hackathon de Modelos Predictivos del PNUD (Bogotá, 6 de octubre de 2026).

Para cada municipio y mes de prueba (2022-10 a 2025-12) el sistema entrega:

1. **Riesgo**: probabilidad de un evento climático con afectación (`prob_evento`).
2. **Magnitud**: personas afectadas estimadas (`personas_desplazadas_estimadas`).
3. **Incertidumbre**: intervalo del 80 % (`personas_desplazadas_q10`, `personas_desplazadas_q90`).

> Este repositorio es **público**. Contiene solo código, configuraciones y documentación. Los datos del reto, el enlace de Drive, las predicciones y los runs nunca se versionan.

## Último CSV de predicciones

El CSV final del equipo es **`/home/hell/TECTON_predicciones_experimentales.csv`** (ruta local, fuera del repositorio; no se versiona).

| Dato | Valor |
|---|---|
| Filas | 42.978 (prueba pública 2022-10 a 2024-04 + prueba oculta 2024-05 a 2025-12) |
| Columnas | `DIVIPOLA, fecha, prob_evento, personas_desplazadas_estimadas, personas_desplazadas_q10, personas_desplazadas_q90` |
| SHA256 | `5f2c2df33d3d797835fb7fdfb49204fba6869f182247bbd4c2d4bc3cbb8db727` |
| Validación | Aprobada con `tecton.schema.validate_predictions` contra las llaves oficiales el 6 de octubre de 2026 |
| Modelo | Variante experimental derivada de la familia final (LightGBM x3 semillas + vecindad espacial + IDEAM + Censo DANE + DNP + bloque econométrico + calibración); ver [docs/ESTADO.md](docs/ESTADO.md) |

Este archivo reemplaza a cualquier `predicciones.csv` de `delivery/` o `colab/` como última versión. Para verificar su integridad: `sha256sum /home/hell/TECTON_predicciones_experimentales.csv`.

---

## 1. Equipo y frentes

| Frente | Responsable | Superficie |
|---|---|---|
| Datos y modelo | Claude Code | `src/tecton/`, `configs/`, `tests/`, notebook generado |
| Modelos econométricos | Laura | Motores adicionales bajo el contrato de `models.py` |
| Estética de datos, gráficas y mapa | Cursor | `dashboard/src/` |
| Pulido de la interfaz | Laura | Jerarquía, copy, estados, responsive y accesibilidad |

Detalle de responsabilidades y contratos entre frentes: [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md).

## 2. El reto en una tabla

| Aspecto | Regla |
|---|---|
| Datos | 1.102 municipios × 96 meses. Entrenamiento 2018-01 a 2022-09 (con targets); prueba pública 2022-10 a 2024-04; prueba privada 2024-05 a 2025-12 |
| Entrega | Un CSV de 42.978 filas y seis columnas fijas: `DIVIPOLA, fecha, prob_evento, personas_desplazadas_estimadas, personas_desplazadas_q10, personas_desplazadas_q90` |
| Calificación | AUC-ROC 50 %, RMSE log1p 20 %, Winkler log1p (α 0.2) 5 %, replicabilidad 5 %, panel de finalistas 20 % |
| Nota automática | Normalizada contra una predicción nula (notebook base, T&C 6.3); máximo 75 puntos. Implementada como `puntos_75` |
| Envíos | Máximo uno cada 30 minutos; cuenta la última entrega válida. Cierre: la diapositiva dice 15:00 y el notebook base 14:00, usar **14:00** hasta aclarar |
| Ejecución | Entrenamiento y validación en **Google Colab** |
| Datos externos | Solo fuentes delimitadas a Colombia, desde 2018 y sin observaciones de los meses de prueba. Decisión del equipo: **ninguno** en el modelo |
| Descalifica | Reconstruir el conjunto oculto, predicciones fijas o a mano, compartir datos fuera del equipo, plagio sin citar, entregar tarde o por otro canal |
| Declarar | Uso de IA generativa en modelo y código (herramienta, versión, uso) y fuentes externas. El geovisor puede hacerse con IA sin declararlo |

Resumen completo y fuentes: [docs/RETO.md](docs/RETO.md). Los Términos y Condiciones oficiales prevalecen.

## 3. Arquitectura

```text
hackathon_datos.zip (Drive PNUD)
        │  URL_DATOS / subida manual en Colab
        ▼
schema.py ── auditoría estricta: filas, 1.102 municipios/mes, DIVIPOLA de 5 dígitos,
        │      fechas AAAA-MM-01, llaves únicas, targets válidos, SHA256 por archivo
        ▼
features.py ── covariables (TerriData, Copernicus DEM, ONI) + calendario
        │       + historial causal suavizado: municipal, estacional, departamental y global.
        │       Entrenamiento: solo meses anteriores. Validación/prueba: historial congelado al corte.
        ▼
models.py ── cuatro cabezas por motor (sklearn HistGB o CatBoost):
        │     clasificador de evento, regresión log1p(personas), cuantiles 0.10 y 0.90
        ▼
pipeline.py ── 5 folds temporales por meses completos + stress de 12 meses
        │       → entrenamiento final hasta 2022-09 → 39 meses de predicción
        ▼
runs/RUN_ID/ ── manifest con hashes, métricas, OOF, predicciones validadas, modelo,
        │        snapshot del código y tabla de replicabilidad
        ▼
promote → state/champion.json ── solo si mejora bajo el mismo protocolo
        ▼
bundle ── CSV de entrega, notebook reproducible, replicabilidad, declaraciones, geovisor
```

Métricas de validación (`metrics.py`): AUC, RMSE log1p sobre todas las filas, Winkler log1p α 0.2, cobertura del intervalo 80 % y `puntos_75`.

Criterios de promoción (operativos, no pruebas de significancia): run completo con cinco folds y stress de 12 meses; frente al champion, +0.002 AUC medio, mejora en ≥ 60 % de folds, pérdida máxima de 0.005 AUC en el último fold y en el stress, RMSE ≤ +3 %, Winkler ≤ +10 % y sin bajar `puntos_75`.

## 4. Flujo de trabajo

**PC local** desarrolla y prueba con datos sintéticos. **Colab** entrena y valida con los datos oficiales. Claude revisa solo resúmenes agregados; nunca filas, targets ni OOF.

```text
PC: editar código → tests + smoke sintético → tecton notebook → subir el .ipynb a Colab
Colab: datos → auditoría → run (first_submission, baseline, catboost…) → compare → promote
Colab → PC: resumen agregado (para revisión) · entrega (formulario) · run sin OOF (geovisor)
```

### En Google Colab

1. Subir `notebooks/tecton_colab.ipynb` y trabajar sobre una copia del equipo.
2. Primera celda:
   - `URL_DATOS`: pegar el enlace de Drive del PNUD (solo en la copia de Colab, nunca en el repo) o subir `hackathon_datos.zip`.
   - `CONFIG_NAME`: `None` (config embebida), `'first_submission'`, `'baseline'` o `'catboost'`.
   - `USE_SYNTHETIC_DATA = True` para un ensayo con datos inventados.
   - `USE_DRIVE = True` para conservar runs si Colab se desconecta.
3. Ejecutar todo. Para otro experimento, cambiar `CONFIG_NAME` y ejecutar desde **Configuración**.
4. Promover con `PROMOTE_RUN_ID`.
5. Descargar:
   - **resumen-*.zip**: métricas y manifiestos sin filas. Dejarlo en `colab/` para revisión.
   - **entrega-RUN_ID.zip**: CSV, notebook, replicabilidad y declaraciones. Va al formulario oficial.
   - **run-RUN_ID.zip**: run sin `oof.csv`, para el geovisor local.

Si la instalación reporta versiones distintas del lock, reiniciar la sesión y ejecutar desde el inicio.

### En la PC (Arch Linux, uv, Python 3.12)

```bash
uv sync --frozen
uv run --frozen python -m tecton doctor
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check src tests scripts

# Ensayo sintético
uv run --frozen python -m tecton synthetic
uv run --frozen python -m tecton run --config configs/smoke.json
uv run --frozen python -m tecton run --config configs/catboost-smoke.json

# Regenerar y validar el notebook de Colab
uv run --frozen python -m tecton notebook --config configs/first_submission.json
uv run --frozen python scripts/check_notebook.py notebooks/tecton_colab.ipynb --execute-python-synthetic

# Traer un run entrenado en Colab y exportar el geovisor (requiere data/raw y dashboard/dist)
uv run --frozen python -m tecton import-run ~/Descargas/run-RUN_ID.zip
uv run --frozen python -m tecton dashboard --run-id RUN_ID --geojson data/geography/municipios.geojson
uv run --frozen python -m tecton bundle --run-id RUN_ID

# Tiempo restante y registro de envíos (no envía nada)
uv run --frozen python -m tecton clock
uv run --frozen python -m tecton record-submission RUN_ID --received-at 2026-10-06T11:30:00-05:00
```

Comandos del harness: `doctor`, `synthetic`, `audit`, `run`, `compare`, `import-run`, `promote`, `bundle`, `dashboard`, `geography`, `notebook`, `freeze`, `clock`, `record-submission`.

### Geovisor

React + Kepler.gl en `dashboard/`, servido en 127.0.0.1 sin servicios externos. Ver [docs/DASHBOARD.md](docs/DASHBOARD.md).

```bash
cd dashboard && npm ci && npm test && npm run build
python3 -m http.server 8765 --bind 127.0.0.1 --directory dashboard/dist
```

## 5. Estructura

| Ruta | Contenido |
|---|---|
| `src/tecton/schema.py` | Contrato de entrada y de entrega, auditoría y hashes |
| `src/tecton/features.py` | Covariables e historial causal/congelado |
| `src/tecton/models.py` | Motores HistGB y CatBoost; punto de integración de nuevos motores |
| `src/tecton/metrics.py` | AUC, RMSE log1p, Winkler, cobertura y `puntos_75` |
| `src/tecton/pipeline.py` | Folds, runs, comparación y promoción |
| `src/tecton/artifacts.py` | Notebook Colab, paquetes de exportación/importación, replicabilidad, bundle |
| `src/tecton/dashboard.py` | Exportador del geovisor y cartografía DANE |
| `configs/` | `first_submission`, `baseline`, `catboost` y sus versiones smoke |
| `tests/` | Invariantes temporales, contrato, métricas, hooks, flujo Colab |
| `notebooks/tecton_colab.ipynb` | Notebook generado con el código embebido; sin datos ni enlace |
| `docs/` | Reto, arquitectura, estado, dashboard, declaraciones de IA y fuentes |
| `dashboard/` | Geovisor React + Kepler.gl |
| No versionados | `data/`, `runs/`, `state/`, `delivery/`, `export/`, `colab/`, `ColabBase/`, `*.local.md` |

## 6. Estado

### Hecho

- [x] Motor reproducible con validación temporal sin fuga, historial congelado y QA de la entrega.
- [x] Contrato verificado con los datos oficiales: auditoría estricta aprobada; diccionario y formato de entrega sin cambios de contrato.
- [x] Notebook Colab generado desde el mismo motor, con carga de datos compatible con el notebook base del PNUD.
- [x] Nota automática oficial (`puntos_75`) en cada fold y en la promoción.
- [x] CatBoost entrenando en la PC; corregido el sesgo de sus cuantiles con muchos ceros.
- [x] Primer run real liviano verificado de punta a punta (validación de 12 meses), ver [docs/ESTADO.md](docs/ESTADO.md).
- [x] Tabla de replicabilidad con un proxy global para cada variable; declaración de fuentes (`docs/fuentes_datos.csv`).
- [x] Geovisor React + Kepler.gl compilado y probado con datos de ensayo.
- [x] 17 tests Python y ruff en verde.

### Pendiente

- [ ] Ejecutar el notebook en Google Colab (instalación del lock, ensayo sintético y datos oficiales).
- [ ] Primera entrega provisional por el formulario y registro con `record-submission`.
- [ ] `baseline` y `catboost` en cinco folds + stress; promover el primer champion.
- [ ] Mejoras de una hipótesis por run: historial causal por tipo de evento, calibración del intervalo al 80 %, interacción ENSO × estacionalidad.
- [ ] Modelos econométricos de Laura integrados como motor y comparados bajo el mismo protocolo.
- [ ] Geovisor con el run real y cartografía DANE; prueba visual de WebGL.
- [ ] Completar `docs/ai_usage.csv` (versión de Claude Code y de Codex, revisión humana).
- [ ] Aclarar con el organizador la hora exacta de cierre (14:00 vs. 15:00).
- [ ] Documento de dos páginas si el equipo queda finalista.

## 7. Reglas de trabajo

- No leer, imprimir ni compartir filas, targets u OOF reales fuera del equipo; trabajar con agregados.
- No editar a mano `runs/`, `delivery/`, `state/` ni predicciones; usar el harness.
- Cambiar una hipótesis por experimento, con los mismos folds y semilla. Un run fallido no reemplaza al champion.
- No elegir modelos por la tabla pública: la nota oficial es la prueba privada.
- No versionar datos, el enlace de Drive, `ColabBase/` ni salidas de notebooks.

Instrucciones para Claude Code: [CLAUDE.md](CLAUDE.md) y `.claude/` (rules, skills, hooks).

## 8. Declaraciones

- IA generativa: [docs/ai_usage.csv](docs/ai_usage.csv).
- Fuentes de datos: [docs/fuentes_datos.csv](docs/fuentes_datos.csv).
- Bibliotecas: numpy, pandas, scikit-learn, CatBoost, React, Vite, Kepler.gl. Cartografía del geovisor: DANE MGN2024.
- Punto de partida metodológico: notebook base de la Hackathon PNUD 2026.

Licencia: [LICENSE](LICENSE).
