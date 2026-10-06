# Estado operativo

- Scaffold inicial: paquete Python, dos motores CPU, cinco folds, stress de 12 meses, cuantiles, QA CSV y notebook portátil.
- No se han recibido CSV oficiales ni se ha medido desempeño competitivo.
- Se verificaron 14 tests Python, 7 tests del dashboard, ruff, motor sklearn en cinco folds con stress y el bundle sintético. Ver docs/VERIFICACION.md.
- CatBoost instalado: entrenamiento pendiente en PC/Colab por acceso /proc restringido aquí. Kernel real de notebook pendiente; las celdas se probaron en un proceso Python separado.
- Los ensayos sintéticos verifican ejecución y contratos, no calidad predictiva.
- Champion real: pendiente. Consultar state/champion.json cuando exista.
- Pendientes antes de las 10:00: instalar en la PC, correr doctor y smoke, identificar los T&C y versión de IA, verificar notebook en Colab si hay tiempo.

## Registro breve de cambios

Registrar fecha, hipótesis, archivos, run_id, métricas agregadas, duración, comandos verificados y decisión. No pegar filas ni OOF reales.

## Geovisor local v1

React + Kepler.gl implementados y compilados; CSV/JSON, filtros compartidos, tendencias, ranking, validación temporal y trazabilidad. Exportador Python y bundle con geovisor comprobados con un run sintético. Datos oficiales, descarga nacional DANE y prueba visual de WebGL en la PC pendientes. Ver docs/DASHBOARD.md. Conservar esta versión como referencia antes de optimizar.
