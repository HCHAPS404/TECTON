---
name: tecton-bootstrap
description: Verificar el entorno local y ejecutar el primer ensayo sintético del proyecto TECTON PNUD antes de recibir datos oficiales.
---

1. Leer CLAUDE.md, README.md y docs/ESTADO.md. No reemplazar una estructura ya funcional.
2. Verificar `uv --version`, Python administrado 3.12 y `uv sync --frozen`. Usar git local; conservar archivos existentes.
3. Ejecutar doctor, unit tests y ruff. Reparar únicamente fallos concretos.
4. Generar synthetic y ejecutar configs/smoke.json y configs/catboost-smoke.json. Si CatBoost está bloqueado por el entorno, registrar el error y conservar el motor sklearn; no declarar entrenamiento CatBoost verificado. Son ensayos funcionales, no evaluaciones competitivas.
5. Generar y validar notebook con el mismo motor. Confirmar dependencias, código incluido y ausencia de datos oficiales.
6. Reportar comandos ejecutados, run_id, tiempo y limitaciones. Actualizar docs/ESTADO.md.
