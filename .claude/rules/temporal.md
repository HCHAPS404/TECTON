---
paths:
  - "src/tecton/features.py"
  - "src/tecton/pipeline.py"
  - "configs/*.json"
  - "tests/*.py"
---
# Temporalidad

- Partir por fecha, no por fila ni KFold aleatorio. Todos los municipios de un mes reciben el mismo corte.
- No usar targets ni columnas de historial del mes objetivo como X. Los agregados temporales son la única vía prevista para targets históricos.
- Excluir todo el mes actual del historial global/departamental. Excluir la fila actual de historia municipal/estacional.
- En validación, no actualizar historia con labels del mismo bloque. En test, congelar en 2022-09.
- Test explícito: cambiar un target futuro o del mes actual no altera X histórico de filas anteriores/del mismo mes; agregar labels a validación no altera transform.
- Tratar DIVIPOLA como string de cinco caracteres. Validar orden y llave al exportar.
