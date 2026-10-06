# Verificación del scaffold

Fecha: 6 de octubre de 2026. Entorno de ejecución disponible: Linux, Python 3.12.14 administrado en un venv, numpy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 y CatBoost 1.2.8 instalado. No es la PC Arch del usuario ni un runtime alojado de Google Colab.

| Comprobación | Resultado |
|---|---|
| Instalación desde pyproject/uv.lock y export de requisitos | Completada |
| doctor | Versiones e imports comprobados |
| 12 tests de invariantes | Pasaron |
| ruff sobre src, tests, scripts y hooks | Pasó |
| sklearn con dos folds sintéticos | Entrenó, predijo, serializó y exportó CSV válido |
| sklearn con cinco folds + stress de 12 meses | Completó todos los modelos y validaciones con 20 municipios inventados |
| Promoción de demo | Completada en champion-demo, sin champion real |
| Bundle de demo | CSV, snapshot, notebook, dashboard, replicabilidad y metadatos generados |
| Notebook nbformat, sintaxis de todas las celdas | Validado |
| Celdas del notebook en proceso Python nuevo | Ejecutadas con datos sintéticos; completó entrenamiento, dashboard y bundle |
| CatBoost entrenamiento | Bloqueado por el sandbox: lectura de /proc/self/statm no disponible |
| Kernel real Jupyter/nbclient | Bloqueado por restricciones de conexiones locales del sandbox |
| Claude Code instalado y skills/hook cargados en la PC | Pendiente; formato contrastado con documentación oficial, hooks de archivos probados mediante stdin JSON |
| Datos reales / validador oficial / score competitivo | Pendiente, no disponibles |

Los tests verifican: excluir targets actuales/futuros del historial, excluir otros municipios del mes actual, historia de test congelada, ignorar targets añadidos a validación, conservar meses enteros, definiciones de métricas, llaves/orden/bounds, marcador sintético, archivo de prueba sin targets, inicializar carpeta del scaffold, promoción comparable, protección de archivos y rechazo de artefactos modificados.

No extrapolar tiempos o AUC sintéticos al dataset oficial. Los parámetros siguen siendo iniciales. No se completó una ejecución alojada en Colab ni el entrenamiento CatBoost aquí.

## Comprobaciones que ejecutar en la PC

```bash
uv run --frozen python -m tecton doctor
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check src tests scripts .claude/hooks
uv run --frozen python -m tecton synthetic
uv run --frozen python -m tecton run --config configs/catboost-smoke.json
uv run --frozen python scripts/check_notebook.py notebooks/tecton_colab.ipynb --execute-synthetic
```

En Colab, abrir el notebook limpio, activar USE_SYNTHETIC_DATA y ejecutar todo. Registrar versión Python y paquetes del manifest. Después usar la plantilla con datos oficiales y USE_SYNTHETIC_DATA=False. Confirmar instrucciones vigentes de la organización sobre plataforma y confidencialidad.

## Verificación del geovisor, 6 de octubre de 2026

- 14 tests Python y ruff: aprobados.
- Instalación npm desde lockfile con npm ci, 7 tests de frontend e integración Kepler: aprobados. Node 24.19.0, npm 11.9.0.
- Compilación Vite de producción: completada. Se resolvieron módulos Node usados por Kepler con polyfills de navegador. Persisten avisos de anotaciones en dependencias y tamaño del chunk Kepler; no son fallos de compilación.
- Probado con Redux/Kepler real, sin GPU: carga de polígonos, DIVIPOLA como texto, reemplazo de datos del mes sin duplicar capas y dominio 0–1.
- Contratos CSV/JSON, fechas, duplicados, límites, intervalos, coordenadas, códigos geográficos y coherencia filtros/KPIs/mapa: probados.
- Exportador real sobre run sintético validado: generó 780 filas y metadatos; preservó hashes y rechazó exportación de ensayo sin --demo.
- Bundle en carpeta aislada: incluyó geovisor, datos.json, HTML de respaldo, notebook, predicciones y replicabilidad.
- Notebook regenerado: nbformat y sintaxis de celdas válidos. La ejecución alojada en Colab sigue pendiente.
- DANE: metadatos y campo mpio_cdpmp contrastados con fuente oficial; descarga directa bloqueada por HTTP 403 en este entorno. No se incluye cartografía nacional ni se acredita cobertura sobre datos reales.
- Navegador local/WebGL, selección gráfica, responsive y teclado: pendientes en la PC. No se inició un servidor de previsualización en este entorno.

El ensayo contiene códigos y valores inventados, sin ubicaciones reales. No mide desempeño competitivo. El geovisor del ZIP abre vacío y permite cargar explícitamente el ensayo. Los hashes de la compilación están en dashboard/RELEASE.json.
