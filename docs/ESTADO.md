# Estado operativo

- Scaffold inicial: paquete Python, dos motores CPU, cinco folds, stress de 12 meses, cuantiles, QA CSV y notebook portátil.
- CSV oficiales recibidos (hackathon_datos.zip, 6 oct 10:00): auditoría estricta aprobada.
- Se verificaron 14 tests Python, 7 tests del dashboard, ruff, motor sklearn en cinco folds con stress y el bundle sintético. Ver docs/VERIFICACION.md.
- CatBoost instalado: entrenamiento pendiente en PC/Colab por acceso /proc restringido aquí. Kernel real de notebook pendiente; las celdas se probaron en un proceso Python separado.
- Los ensayos sintéticos verifican ejecución y contratos, no calidad predictiva.
- Champion real: pendiente. Consultar state/champion.json cuando exista.
- Pendientes antes de las 10:00: instalar en la PC, correr doctor y smoke, identificar los T&C y versión de IA, verificar notebook en Colab si hay tiempo.

## Registro breve de cambios

Registrar fecha, hipótesis, archivos, run_id, métricas agregadas, duración, comandos verificados y decisión. No pegar filas ni OOF reales.

## Geovisor local v1

React + Kepler.gl implementados y compilados; CSV/JSON, filtros compartidos, tendencias, ranking, validación temporal y trazabilidad. Exportador Python y bundle con geovisor comprobados con un run sintético. Datos oficiales, descarga nacional DANE y prueba visual de WebGL en la PC pendientes. Ver docs/DASHBOARD.md. Conservar esta versión como referencia antes de optimizar.

## Adecuación a Colab y reglas del día, 6 de octubre 09:55

- Notebook generado (`tecton notebook`): selector CONFIG_NAME, carga por URL_DATOS/hackathon_datos.zip como el notebook base, compare/promote en Colab, Drive opcional, descargas separadas: resumen agregado (review_pack), entrega (bundle) y run sin OOF para la PC (run_pack + `tecton import-run`).
- Métrica puntos_75 con la normalización del notebook base; promoción rechaza bajar puntos_75. Protocolo de métricas v2. Tiempos por fold en metrics.json. threads=2 en configs.
- clock: 14:00 límite seguro y 15:00 de la diapositiva (contradicción pendiente de aclarar).
- CatBoost entrenó en la PC (antes bloqueado en el sandbox). Defecto corregido: cuantiles con leaf_estimation_method=Exact daban cobertura ~0.06 con ceros masivos; con Gradient catboost-smoke da cobertura 0.67 (sintético). sklearn smoke: 0.89.
- Verificado: 17 tests, ruff, smoke sklearn, catboost-smoke, notebook ejecutado completo en proceso Python con sintéticos. Pendiente: ejecución real en Colab (instalación del lock), datos oficiales.
- Siguiente hipótesis tras baseline: calibración del intervalo hacia 80% con OOF.

## Datos oficiales, 6 de octubre 10:05

- Auditoría estricta OK: filas, municipios y fechas coinciden con la guía. ZIP trae también diccionario_datos.csv y formato_entrega.csv, sin cambios de contrato.
- Hallazgos agregados de los datos: solo en docs/hallazgos_datos.local.md (no versionado; el repo es público).
- Ideas priorizadas: historial causal por tipo de evento, calibración de intervalo, interacción ENSO x estacionalidad municipal.
