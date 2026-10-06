---
name: tecton-audit
description: Auditar el contrato de los CSV PNUD y la integridad temporal antes de entrenar o exportar. Usar cuando se reciban los datos oficiales o cambie el pipeline.
---

1. Ejecutar `uv run --frozen python -m tecton audit --config configs/baseline.json`.
2. Leer únicamente el resumen agregado. No usar Read sobre data/raw ni cat/head/print de filas reales.
3. Verificar 62814/20938/22040 filas, fechas, 1102 municipios por mes, cinco dígitos DIVIPOLA, targets válidos y nulos de covariables.
4. Revisar las definiciones del diccionario oficial cuando esté disponible. Si cambia el contrato, corregir código explícitamente y dejar constancia; no saltar la auditoría con strict=false.
5. Ejecutar tests de historia causal y exportación si cambiaron esas partes. Registrar hallazgos y siguiente acción.
