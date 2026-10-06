# TECTON PNUD

Tu frente es datos y modelo. Antes de editar, leer `docs/ARQUITECTURA.md`. No implementes la estética de gráficas y mapas: eso es de Cursor. No pulses la interfaz: eso es de Laura.

Construir y mejorar un sistema tabular municipio-mes para la Hackathon PNUD del 6 de octubre de 2026. Entregar CSV válido, código reproducible y tabla de replicabilidad. Leer `docs/RETO.md` y `docs/ESTADO.md` al comenzar. Los T&C oficiales prevalecen sobre el resumen disponible.

## Arquitectura y comandos

- Python local administrado con uv, versión 3.12. CPU por defecto.
- Lógica única en `src/tecton/`; configs JSON en `configs/`; notebook generado, no escrito como un motor distinto.
- `uv run --frozen python -m tecton doctor`
- `uv run --frozen python -m unittest discover -s tests -v`
- `uv run --frozen ruff check src tests scripts`
- `uv run --frozen python -m tecton synthetic`
- `uv run --frozen python -m tecton run --config configs/smoke.json`
- `uv run --frozen python -m tecton compare`

## Reglas permanentes

1. Conservar CSV originales; procesarlos localmente mediante código. No imprimir ni pegar filas, targets u OOF reales en conversaciones. Usar auditorías agregadas; confirmar alcance permitido de compartir agregados en los T&C.
2. Usar fechas completas para folds. Historial estrictamente anterior a cada fila de entrenamiento; validación y test usan historia congelada en su corte.
3. Predecir `tiene_evento`, `log1p(personas_desplazadas)` y cuantiles 0.10/0.90. Mantener las métricas exactas. No inventar el score normalizado de los T&C ausentes.
4. Baseline sklearn primero; CatBoost después. Conservar CPU/semilla/folds y cambiar una hipótesis por experimento. Un run fallido no reemplaza el champion.
5. No editar a mano `runs/`, `delivery/`, `state/champion.json` o predicciones. Usar el harness para escribir artefactos y promover.
6. El notebook de entrega se genera del snapshot de código usado para entrenar el run elegido.
7. No añadir datos de emergencias de los períodos de prueba ni reconstruir targets. Rama de clima adicional desactivada hasta aclaración del organizador.
8. No instalar frameworks de agentes, Spark, Docker, MLflow ni modelos neuronales para esta jornada. Una sesión de Claude administra código; el entrenamiento lo administra Python.
9. Al terminar una unidad de trabajo: reportar cambio, comandos verificados, limitaciones y próximo paso; actualizar `docs/ESTADO.md`. No declarar pruebas no ejecutadas.
10. Versionar solo código, configs e instrucciones con git local. No crear remotos ni publicar sin instrucción explícita.

Las rules detallan invariantes; las skills ejecutan flujos. Los hooks protegen herramientas de edición, pero no son una barrera completa contra procesos Bash. La integridad final la comprueban los validadores del harness.
