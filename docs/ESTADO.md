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

## Nuevo champion, 6 oct 12:10

- 20261006T165706912949Z-lgbm-espacial-ideam: LightGBM (800 it, lr 0.01, 15 hojas, submuestreo) x3 semillas + vecindad espacial k=8 (coordenadas DANE) + climatología IDEAM 2018-01 a 2022-09 + calibración.
- 5 folds: puntos 24.34, AUC 0.7238, RMSE 1.274, Winkler 3.216. Stress 12 m AUC 0.7418. Stress lejano 13-24 m AUC 0.6955, puntos 21.02 (base 19.90).
- vs baseline-calibrado (bootstrap por meses): ΔAUC +0.0078 [+0.004, +0.013]; ΔWinkler −0.22; Δpuntos +1.12 [+0.74, +1.65], P=1.00; mejora en 5/5 folds. Promovido por el harness.
- Ablación rápida (clasificador, objetivo folds+lejano): hist 0.6991; LightGBM 0.7054; +coords 0.7066; +vecindad 0.7070; +IDEAM 0.7081; +vecindad+IDEAM 0.7094. Vecindad k=5 0.7079. Optuna (hist 40 pruebas: 0.7029; LightGBM 11 pruebas: sin mejora sobre 0.7054).
- Notebook Colab con esta config por defecto, verificado en entorno tipo Colab (instala lightgbm; datos externos embebidos).

## Ronda de clima y vulnerabilidad, 6 oct 12:00-12:50

Tabla pública: champion lgbm-espacial-ideam = 27.6 (AUC 0.757, RMSE 1.171, Winkler 2.66). Líder 28.4 (AUC 0.764); 2.º y 3.º con AUC 0.761.

| Variante (evaluación rápida sobre champion 0.7094) | Objetivo | Decisión |
|---|---|---|
| Perfil estacional de lluvia (total anual + fracción mensual) | 0.7124 | adoptado |
| + intensidad IDEAM (máx 10 min, fracción de intervalos con lluvia) | 0.7136 | adoptado en x11 |
| + historial de viviendas/tipos de evento (vulnerabilidad) | 0.7142 | x11 |
| k=15, suavizado 10, más árboles, sin n_estaciones, combinado de mejoras pequeñas | ≤ 0.7135 | descartados (ruido) |
| SGC inventario de movimientos en masa | — | no tabular en la API |

Runs completos:
- 20261006T171507194824Z-lgbm-ideam-perfil: puntos 24.59; stress 12 m AUC 0.7447; lejano 0.7015 / 21.67. vs champion anterior Δpuntos +0.29 [+0.17, +0.44], P=1.00. Promovido y enviado 12:40.
- 20261006T172320268707Z-lgbm-clima-vulnerabilidad (x11): puntos 24.63; stress 12 m 0.7413; lejano 0.7038 / 21.95. vs perfil: empate (P=0.56).
- Mezcla OOF 50/50 perfil + x11: 24.70 (AUC 0.7268). Implementado como `ensemble` reproducible en el pipeline; run completo en curso.
- Límite de entrega confirmado: 14:45.

## Motor econométrico, 6 de octubre

- Familia `econ` en `configs/econ.json`: valla cloglog, medias de Mundlak, cuantiles de la mezcla con ceros. Run `20261006T155655465037Z-econ-hurdle-mundlak`.
- Cortes: entrenamiento 2018-01 a 2022-09; prueba pública 2022-10 a 2024-04 con historia congelada; prueba privada 2024-05 a 2025-12 con historia de entrenamiento real más la predicción pública. No se usaron reportes UNGRD desde 2022-10.
- Fuentes: CSV oficial; DANE PPED-AreaMun-2018-2042 (cobertura DIVIPOLA 1.000, años 2018-2022); UNGRD wwkg-r6te (2019-01 a 2022-09, cobertura 0.917). Citas en `docs/fuentes_datos.csv`.
- Validación agregada: AUC medio 0.668, RMSE-log 1.286, Winkler 3.321, cobertura 0.922, puntos_75 18.4. Estrés 12 meses: AUC 0.691, cobertura 0.922. No promovido.
- GARCH(1,1) de panel (`src/tecton/garch.py`): varianza condicional de log1p personas por municipio; alfa y beta comunes estimados por cuasi-verosimilitud solo con meses <= corte; después del corte, pronóstico a h meses (mismo para pública y privada). Test de fuga: la varianza inicial usaba el máximo de todos los meses; corregido.
- Integrado al modelo general: `features.econ` (GARCH + rezagos de conteos oficiales por tipo + UNGRD wwkg-r6te 2019-01 a 2022-09 + log población DANE 2018-2022) en `Features`, con hashes en el manifest. Config `configs/lgbm-espacial-ideam-econ.json` = champion + este bloque (una hipótesis).
- PC, mismos folds y semillas, champion re-ejecutado localmente (reproduce stress 12 m 0.7418 y lejano 0.6955):

| run | puntos_75 | AUC | RMSE-log | Winkler | cobertura | stress 12 m | lejano 13-24 m |
|---|---|---|---|---|---|---|---|
| lgbm-espacial-ideam | 24.34 | 0.7238 | 1.274 | 3.216 | 0.929 | 26.20 (AUC 0.7418) | 21.02 (AUC 0.6955) |
| + bloque econométrico | 24.61 | 0.7266 | 1.275 | 3.203 | 0.930 | 26.38 (AUC 0.7438) | 21.12 (AUC 0.6972) |

- AUC por fold (econ − champion): +0.0045, +0.0053, +0.0003, −0.0007, +0.0043. Pendiente: bootstrap pareado (`scripts/stat_compare.py`) y promoción por harness; corre ~2x más lento (≈4 min por fold en la PC).
- EQTransformer no se implementó: es una red neuronal de detección sísmica sobre formas de onda (regla 8 prohíbe modelos neuronales y no aplica a un panel municipio-mes). Sustituto admisible si se quiere la señal: catálogo sísmico del SGC 2018-01 a 2022-09 como rezago municipal.
- Reestimado con CatBoost (DIVIPOLA categórica, 300 iteraciones). Run `20261006T160711740047Z-econ-panel-boost`. AUC medio 0.701, RMSE-log 1.276, Winkler 3.555, cobertura 0.720, puntos_75 21.6. Estrés 12 meses: AUC 0.738, RMSE-log 1.249. El corte 2021-10 a 2022-03 llegó a AUC 0.782 y RMSE-log 1.091. El corte más reciente, 2022-04 a 2022-09, quedó en AUC 0.687. No promovido.

## Cierre, 6 oct 14:42

Envíos en tabla pública (puntos / AUC / RMSE / Winkler):
- baseline-calibrado 26.2 / 0.748 / 1.177 / 2.90
- bagging 26.6 / 0.751 / 1.176 / 2.86
- lgbm-espacial-ideam 27.6 / 0.757 / 1.171 / 2.66
- lgbm-ideam-perfil 27.7 / 0.757 / 1.171 / 2.65
- lgbm-perfil-exposicion 27.8 / 0.758 / 1.170 / 2.65
- final-censo-econ 28.4 / 0.763 / 1.168 / 2.63
- **final-dnp (entrega final)** AUC 0.764

Entrega final: 20261006T184216080776Z-final-dnp (LightGBM x3 semillas + vecindad espacial + IDEAM climatología y perfil + Censo DANE 2018 + capacidades DNP + bloque econométrico de Laura + calibración). vs final-censo-econ: ΔAUC +0.0007 (P=0.98), Δpuntos +0.08 (P=0.99); el harness no lo promueve (umbral +0.2), se envió por evidencia estadística consistente. Champion del harness: final-censo-econ.
Validación: 5 folds puntos 25.17, AUC 0.7311; stress 12 m AUC 0.7468; lejano 13-24 m AUC 0.7028 / 21.84 (primer envío: 19.90).
Descartados al cierre: historial UNGRD 2014-2015 (evaluación +0.0004 dentro del ruido; run corto peor en fold 2022-04/09: 0.7068 vs 0.7112), Optuna sobre variables finales (sin mejora en 2 pruebas, detenido por tiempo).
Bundle: delivery/20261006T184216080776Z-final-dnp (CSV, notebook desde snapshot, replicabilidad 97 variables, fuentes, declaración de IA, geovisor). Notebook del equipo con enlace: colab/tecton_colab_equipo_FINAL.ipynb (no versionado).

## Último intento, 6 oct 14:53

- Run `20261006T193544677739Z-cierre-dnp-hist` (final-dnp + historial UNGRD 2014-2017 mensual) quedó sin terminar: sin predicciones ni métricas. No entregable.
- Mezcla 50/50 final-dnp + final-censo-econ (`scripts/blend_runs.py`, OOF 5 folds): AUC 0.7309 vs 0.7311 de final-dnp solo; puntos 25.13 vs 25.17. No mejora. CSV generado en `colab/mezcla-20261006T195227Z/` pero descartado.
- Decisión: la entrega vigente sigue siendo `20261006T184216080776Z-final-dnp` (AUC pública 0.764). No hubo tiempo para un run completo nuevo antes de las 14:58.

## Último CSV designado por el equipo

- Archivo: `entrega/TECTON_predicciones_experimentales.csv` (versionado por instrucción del equipo para el despliegue local de Laura; copia original en `/home/hell/`). SHA256 `5f2c2df3…db727`, 42.978 filas. Generado con ChatGPT Codex (modo work) sobre una copia del proyecto.
- Validado con `validate_predictions` contra las llaves de `prueba_equipos.csv` + `prueba_oculta.csv`: aprobado. Agregados: prob media 0.089, punto medio 0.49 en log1p, q10 ≤ q90 en todas las filas.
- Su hash no coincide con ningún `runs/*/predicciones.csv`, ni con los bundles de `delivery/`, ni con las mezclas de `colab/`: proviene de una variante experimental generada fuera del harness local. Se documenta en README como última versión del CSV.
- Commit y push a `origin/colab-v1` el 6 de octubre de 2026 por instrucción explícita del equipo.
