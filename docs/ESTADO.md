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

## Experimentos con datos oficiales (PC, mismos 5 folds + stress, semilla 42), 6 oct 10:35

| run | puntos_75 | AUC | RMSE-log | Winkler | cobertura | decisión |
|---|---|---|---|---|---|---|
| baseline-hist | 22.66 | 0.7154 | 1.279 | 3.812 | 0.902 | referencia |
| baseline-calibrado (punto lineal + intervalo con umbral de probabilidad) | 23.17 | 0.7154 | 1.277 | 3.437 | 0.905 | mejor; config por defecto del notebook |
| + historial por tipo de evento | 22.45 | 0.7127 | 1.278 | 3.784 | 0.904 | descartado (sin umbral; baja AUC) |
| catboost-calibrado | — | fold 1: 0.6355 | — | — | — | detenido: 194 s/fold y peor que hist en fold 1 |

- Promoción ahora por puntos_75 (+0.2) con salvaguardas de AUC; protocolo de métricas v3.
- Notebook: sin modo sintético, sin versiones fijas (usa las de Colab e instala solo lo faltante), descarga directa del ZIP por URL_DATOS. Ejecutado completo en un entorno tipo Colab (Python 3.12, numpy 2.0.2, pandas 2.2.2, scikit-learn 1.6.1): mismas métricas que la PC.
- Copia del equipo con el enlace: colab/tecton_colab_equipo.ipynb (no versionada).

## Ronda de mejora del AUC, 6 oct 11:00-11:35 (PC, datos oficiales)

Tabla pública tras primer envío (baseline-calibrado): 3.º de 12, 26.2 puntos, AUC 0.748, RMSE 1.177, Winkler 2.90. Solo referencia; se decide por validación.

Diagnóstico: AUC 0.738 a 1-12 meses del corte y 0.687 a 13-24 meses (la prueba privada está a 20-39). Oráculo con tasa municipal real del período futuro: 0.839; el historial disponible ordena municipios con AUC ~0.665. El techo lo pone la información, no el algoritmo. Importancia: mes x departamento, ingresos, ONI x pendiente, elevación.

| Hipótesis | Resultado | Decisión |
|---|---|---|
| Bagging 3 semillas, max_features 0.7 | 12m 25.95 pts (AUC 0.740); 13-24m 19.94 | mejor |
| Regularización (min_samples_leaf 100, l2 5) | 5 folds 23.21 vs 23.17 | combinar |
| Historial ENSO, estacionalidad departamental, tipos de evento | empate o peor | descartado |
| Más iteraciones, excluir 2018 | peor | descartado |
| Entrenamiento con cortes congelados (h24) | 13-24m AUC 0.660 vs 0.687 | descartado |
| Quitar covariables temporales, suavizado 60, mezcla con logística | igual o peor | descartado |
| SMOTE / pesos de clase | no aplica: AUC es de ranking | no se aplica |

Nueva métrica en runs: stress_far_13_24 (entrena hasta 2020-09, valida 2021-10 a 2022-09). Siguiente: datos externos colombianos desde 2018 (climatología IDEAM), como experimento separado.

## Skills de estadística aplicadas, 6 oct 11:30

- Nuevas skills: `tecton-estadistica` (bootstrap pareado por meses, criterios de decisión, deriva, disciplina experimental) y `tecton-datos-externos` (admisibilidad y protocolo). Scripts: `scripts/stat_compare.py`, `scripts/drift_check.py`.
- bagging vs baseline-calibrado: ΔAUC +0.0002 [−0.0025, +0.0029]; ΔWinkler −0.037 (P=1.00); Δpuntos +0.07 (P=0.67). Empate en riesgo.
- bagging+regularizado vs baseline-calibrado: ΔAUC +0.0018 (P=0.92); Δpuntos +0.14 (P=0.87); stress lejano 19.56 vs 19.90. Harness rechaza (+0.2 requerido).
- Deriva: covariables municipales estables (PSI < 0.1); ONI PSI 1.33. ENSO: entrenamiento 28% Niño/42% Niña; pública 58% Niño; privada 85% Neutral, 0% Niño. La nota oficial dependerá de estacionalidad y perfil municipal más que de ENSO.
- Champion: 20261006T154005442685Z-baseline-calibrado (run de Colab enviado). Envío registrado con hora aproximada 10:50, pendiente de confirmar.
- En curso: climatología municipal de lluvia IDEAM 2018-01 a 2022-09 (fuente externa colombiana); integración `features.external_files` lista.
