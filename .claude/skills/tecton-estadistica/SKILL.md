---
name: tecton-estadistica
description: Análisis estadístico riguroso de experimentos TECTON (nivel posgrado en ciencia de datos). Usar antes de promover un modelo, al comparar runs, al diagnosticar por qué una métrica no mejora o al evaluar deriva entre entrenamiento y prueba.
argument-hint: "run_id_A run_id_B, o 'diagnostico', o 'deriva'"
---

Objetivo: $ARGUMENTS

Actuar como analista con formación doctoral en estadística aplicada y gestión de datos. Cada decisión se apoya en evidencia cuantificada, con incertidumbre, sin mirar filas reales en el chat. Todo cálculo corre en scripts locales que imprimen agregados.

## 1. Preguntas que se responden antes de tocar el modelo

1. **¿Qué mide la nota?** `puntos_75` = 100·(0.5·(AUC−0.5)/0.5 + 0.2·(1−RMSE/RMSE₀) + 0.05·(1−W/W₀)). Calcular cuántos puntos aporta cada término; atacar el que domina (hoy AUC: +0.01 AUC ≈ +1 punto).
2. **¿Qué horizonte importa?** Prueba pública: 1–19 meses tras el corte; privada (nota oficial): 20–39. Reportar siempre folds (1–6 m), stress 12 m y `stress_far_13_24`. Una mejora que solo aparece a horizonte corto no es evidencia para la nota oficial.
3. **¿Dónde está el techo?** Descomponer AUC en ranking entre municipios (dentro de cada mes) y entre meses. Comparar con oráculos diagnósticos (tasa municipal real del período). Si el modelo ya está cerca de lo que la información permite, cambiar de palanca (información nueva), no de algoritmo.

## 2. Protocolo de comparación (A vs B)

Ejecutar `uv run --frozen python scripts/stat_compare.py RUN_A RUN_B`:

- Bootstrap pareado **por bloques de mes** (los municipios de un mismo mes no son independientes): remuestrear meses de validación con reemplazo, B ≥ 1000. Reporta ΔAUC, ΔRMSE, ΔWinkler y Δpuntos con IC 95 % y P(Δ > 0).
- Deltas por fold y conteo de folds que mejoran.

Criterios de decisión (todos):
- Δpuntos medio > 0 y P(Δpuntos > 0) ≥ 0.80 en el bootstrap por meses.
- Mejora en ≥ 60 % de folds y sin pérdida > 0.005 AUC en stress 12 m ni en stress lejano 13–24 m.
- Si el IC 95 % de ΔAUC cruza 0 con amplitud > 0.01, declarar **empate** y preferir el modelo más simple o más estable.
- No cambiar umbrales después de ver el resultado.

## 3. Disciplina experimental

- Una hipótesis por run; mismos folds, semilla y datos (hash). Registrar en `docs/ESTADO.md` hipótesis, run_id, deltas con IC y decisión, también los fracasos.
- Corrección por comparaciones múltiples: tras k variantes sobre los mismos folds, el mejor de k está sesgado al alza. Confirmar la ganadora en un horizonte no usado para elegirla (stress lejano) antes de promover.
- Presupuesto: una búsqueda acotada (≤ 10 variantes por ronda) con hipótesis motivadas por el diagnóstico, nunca un barrido ciego.
- Tamaño del efecto antes que significancia: < 0.002 AUC es ruido operativo.

## 4. Diagnósticos obligatorios cuando una métrica no avanza

- **Deriva** (`uv run --frozen python scripts/drift_check.py`): validación adversaria entrenamiento vs prueba (AUC del clasificador que los distingue) y PSI por covariable. PSI > 0.25 o AUC adversario > 0.75 indica que el modelo extrapola en esa variable.
- **Importancia por permutación** en el horizonte relevante, no en entrenamiento.
- **Calibración del intervalo**: cobertura empírica vs 80 % y Winkler por tramo de probabilidad.
- **Residuos de magnitud**: RMSE en filas con evento vs sin evento.

## 5. Técnicas que no se aplican y por qué

- **SMOTE / sobremuestreo / pesos de clase**: AUC es invariante al balance; inventar municipio-meses rompe la estructura temporal y descalibra probabilidades.
- **KFold aleatorio, target encoding sin corte temporal, features del mes objetivo**: fuga.
- **Elegir por la tabla pública**: la nota oficial es otro período.

## 6. Criterios para actuar de golpe

Cuando el diagnóstico es claro, implementar completo en una sola pasada: opción de config apagada por defecto, test de fuga si toca features, run con protocolo completo, comparación estadística, registro, commit. Si el resultado no supera los criterios, descartar en el mismo ciclo y documentar. No dejar experimentos a medias.
