# Prompt para continuar el dashboard con Claude Code

```text
Continúa el geovisor local de TECTON. Antes de editar, lee CLAUDE.md,
.claude/rules/dashboard.md, docs/DASHBOARD.md y docs/VERIFICACION.md.

Objetivo: presentar los resultados municipales-mensuales del modelo del reto
con React y Kepler.gl, manteniendo Python como única fuente de entrenamiento,
validación y CSV oficial. Debe poder servirse con Python local y cargar el CSV
generado por el mismo motor en Colab.

Inspecciona el estado real del proyecto. Conserva la versión que funciona.
Si existen datos y run real, identifica su run_id y usa el exportador, sin editar
predicciones, manifiestos ni métricas. Si faltan, mantén el estado vacío y una
demostración claramente identificada. No inventes ubicaciones ni desempeño.

Prioridades: comprobar el visor con cartografía DANE en la PC; resolver únicamente
fallos comprobados; verificar conteos, filtros, escala y selección en el mapa;
guardar un resultado reproducible. Después plantea una mejora a la vez con
criterio de aceptación y comparación contra la versión anterior.

No migres el stack, no añadas backend o base de datos y no actualices paquetes
sin una necesidad demostrada. No publiques datos o predicciones. No entrenes
desde filtros ni construyas targets con información del período de prueba.

Validación: npm ci, npm test, npm run build, pruebas Python y ruff; prueba visual
local de WebGL, escritorio/móvil, carga de archivos, cambio de mes, selección
de municipio, filtros, descarga y errores. Si una verificación no se pudo hacer,
registra esa limitación y no la declares aprobada.

Al terminar, reporta cambios, comandos comprobados y pendientes concretos.
```
