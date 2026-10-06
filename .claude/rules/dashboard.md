---
paths:
  - "dashboard/**"
  - "src/tecton/dashboard.py"
  - "docs/DASHBOARD.md"
---

# Invariantes del geovisor

Cursor implementa la estética de datos, gráficas y mapas. Laura pule esa interfaz. Claude no cambia el aspecto; si el exportador Python gana un campo, lo documenta en `docs/DASHBOARD.md` y en `docs/ARQUITECTURA.md`.

- Python es la única fuente de entrenamiento, métricas y CSV oficial. La UI consume un snapshot; no reentrena al filtrar.
- DIVIPOLA siempre es texto de cinco dígitos, también dentro de Kepler. Usar el contrato explícito de mapDataset; no inferencia numérica.
- No cargar ni publicar datos oficiales en servicios externos. Servir en 127.0.0.1. Mantener el fondo local sin tiles ni geocodificación externa.
- GeoJSON WGS84, unión exacta por DIVIPOLA, sin posiciones inventadas ni uniones aproximadas por nombre.
- Un mes por mapa. El mismo filtrado alimenta mapa, ranking, tabla y KPIs; los municipios sin geometría siguen en las cifras y se informan.
- El detalle temporal muestra todos los meses del municipio y debe indicarlo. Las métricas de validación son del run completo y no del filtro geográfico.
- Probabilidad tiene dominio de color 0–1. Magnitud/amplitud usan log1p y dominio del archivo completo con leyenda en personas.
- No confundir personas afectadas reportadas con desplazamiento observado. No sumar cuantiles ni llamar al umbral alerta oficial.
- Los ensayos se identifican como inventados. CSV aislado no acredita procedencia ni métricas. No presentar score oficial a partir de la UI.
- Conservar el HTML de respaldo. WebGL fallido no debe bloquear tabla o gráficos. Sin datos, mostrar estado vacío; error de carga no debe reemplazar el archivo válido.
- Congelar una versión comprobada antes de mejorarla; mantener lockfiles. npm test/build y prueba visual local antes de entregar; informar verificaciones no disponibles.
