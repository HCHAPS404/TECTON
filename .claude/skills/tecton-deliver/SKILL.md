---
name: tecton-deliver
description: Preparar una entrega local revisable del run seleccionado de TECTON; nunca enviar ni publicar automáticamente.
argument-hint: "run_id a exportar o champion"
disable-model-invocation: true
---

Destino: $ARGUMENTS

1. Confirmar que el run tiene status complete y qa_passed, y que no es sintético para una entrega oficial.
2. Usar bundle con el champion o --run-id para una primera entrega provisional.
3. Verificar archivos: predicciones.csv, tecton_colab.ipynb, código snapshot, dashboard, replicabilidad, métricas y declaración de IA.
4. Leer el resumen de QA, no filas reales. Completar docs/ai_usage.csv con herramienta/version reales y revisión del equipo; no inventarlas.
5. Revisar equivalencias pendientes de la tabla de replicabilidad. Usar clock y respetar los 30 minutos entre entregas.
6. Reportar ruta local y pendientes. El usuario envía por el formulario y registra su hora mediante record-submission. No publicar ni transmitir datos.
