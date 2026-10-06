---
name: tecton-datos-externos
description: Evaluar, construir e integrar una fuente de datos externa en TECTON respetando las reglas del reto (Colombia, desde 2018, nada de los meses de prueba). Usar cuando el diagnóstico estadístico indique que el techo es de información.
argument-hint: "fuente candidata"
---

Fuente: $ARGUMENTS

## 1. Filtro de admisibilidad (si falla uno, se descarta)

1. Entidad o cobertura delimitada a Colombia (IDEAM, DANE, IGAC, SGC, DNP, datos.gov.co). Las globales recortadas se desaconsejaron por el organizador.
2. Todas las observaciones usadas con fecha ≥ 2018-01-01.
3. Nada observado después de 2022-09 (inicio de la prueba). Variables mensuales solo como agregados del período de entrenamiento (climatologías, medias), nunca valores del mes predicho.
4. No son reportes de emergencias ni permiten reconstruir targets.
5. Licencia abierta y descargable de forma reproducible por script.

## 2. Valor esperado antes de construir

- ¿Qué parte del techo ataca? (ranking municipal, estacionalidad, magnitud). Debe relacionarse con lo que el diagnóstico mostró como limitante.
- ¿Ya está representada? (p. ej. la elevación ya viene en el dataset).
- Costo en tiempo vs. minutos al cierre. Si no cabe con protocolo completo y margen, no se empieza.

## 3. Construcción

- Script reproducible en `scripts/` que descarga, agrega y escribe `data/external/<fuente>.csv` + `.meta.json` (URL, ids, fecha de descarga, rango temporal, método de agregación e imputación, cobertura). `data/` no se versiona.
- Llave exacta `DIVIPOLA` de 5 dígitos (y `mes` si aplica). Unión por nombre solo normalizada y con desempate por coordenadas; documentar imputaciones.
- Cobertura total de los 1.102 municipios del reto; marcar imputados.

## 4. Integración y evaluación

- Opción de config apagada por defecto (`features.external`), con el hash del archivo en el manifest.
- Un solo cambio frente al mejor modelo vigente; protocolo completo (5 folds, stress 12 m, stress lejano 13–24 m) y `tecton-estadistica` para decidir.
- Agregar la fuente a `docs/fuentes_datos.csv` y a la tabla de replicabilidad con su proxy global. El notebook de Colab debe poder reconstruirla o incluirla.
