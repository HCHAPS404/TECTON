---
paths:
  - "configs/*.json"
  - "src/tecton/models.py"
  - "src/tecton/metrics.py"
  - "docs/ESTADO.md"
---
# Experimentos

- Conservar config, seed, protocolo temporal y hashes de datos por run.
- Cambiar una hipótesis; no mezclar nuevo modelo, nuevas fuentes y nuevo split en una comparación.
- Comparar AUC medio, por fold, último fold y stress de 12 meses; RMSE y Winkler en log1p.
- No editar métricas para hacer pasar promoción. Los umbrales operativos no son significancia estadística ni garantía sobre prueba oculta.
- Hacer primero una semilla. Repetir mejores candidatos con otra semilla solo si queda presupuesto de tiempo.
- Probar CatBoost y baseline antes de añadir un challenger. Evaluar un ensemble mediante OOF comunes si se implementa; el scaffold inicial no contiene un ensemble.
- Ante un fallo, arreglarlo sin reemplazar el CSV ni snapshot del champion.
