---
name: tecton-experiment
description: Ejecutar una mejora controlada de un modelo TECTON conservando el champion y comparando el mismo protocolo temporal.
argument-hint: "hipótesis concreta o ruta de una configuración"
---

Hipótesis solicitada: $ARGUMENTS

1. Leer docs/ESTADO.md, champion y métricas existentes; no asumir que el leaderboard selecciona modelos.
2. Copiar una config a un nuevo nombre y cambiar una hipótesis. Mantener folds, seed y hashes de datos comparables.
3. Ejecutar run, conservar el run_id y comparar con el champion. Ejecutar tests específicos si hubo código nuevo.
4. Revisar todos los folds y stress de 12 meses. Registrar ganancia, costo y degradaciones.
5. No reemplazar champion ni editar predicciones. Sugerir un candidato para promote si pasa los criterios del harness.
6. Si ya hay FROZEN.json, los experimentos quedan separados y no cambian la entrega.
