# Geovisor y dashboard local

La interfaz está construida para explorar el resultado del modelo de la hackathon, con una fila por municipio y mes. La guía solicita geovisor **o** dashboard; esta implementación combina ambos. No contiene datos oficiales ni un modelo competitivo validado. La cartografía oficial debe descargarse o cargarse en la PC antes de presentar el mapa.

## Stack y decisiones

| Capa | Implementación | Motivo |
|---|---|---|
| Entrenamiento y validación | Python, pandas, sklearn/CatBoost del scaffold | Un mismo motor en Arch y Colab |
| Exportación | `tecton dashboard`, CSV oficial y JSON con hashes | Evita volver a entrenar desde la interfaz |
| Aplicación | React 18.3.1 y Vite 7.1.9 | Interfaz propia, filtros compartidos y compilación estática |
| Mapa | Kepler.gl 3.2.6, Redux 4.2.1, MapLibre 3.6.2 | Polígonos municipales, tooltip y selección |
| Gráficos | SVG en React | Tendencias e intervalos sin servicios adicionales |
| Geometrías | GeoJSON WGS84 de DANE, MGN2024 | Unión exacta por DIVIPOLA, sin coordenadas inventadas |
| Servidor local | `python3 -m http.server`, solo 127.0.0.1 | Abrir la versión compilada sin Node ni backend adicional |
| Reproducibilidad | `package-lock.json`, `uv.lock`, manifiestos y tests | Versiones y trazabilidad de cada exportación |

No se requiere base de datos para las 42.978 predicciones de la entrega. El archivo permanece en memoria del navegador y se filtra por mes. Las geometrías se guardan una sola vez, sin replicarlas para todos los meses. Node se requiere únicamente para modificar y recompilar la interfaz. La instalación npm usa `legacy-peer-deps` porque Kepler incluye dependencias antiguas de interfaz con rangos de peers incompatibles con React 18; por eso se conserva el lockfile y se prueba la integración. No actualizar paquetes durante la sesión final sin repetir esas pruebas.

## Abrir ahora, con Python

Desde la raíz del repositorio:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory dashboard/dist
```

Abrir **http://127.0.0.1:8765**. No abrir `index.html` mediante `file://`: los módulos y la carga del JSON necesitan HTTP. El visor arranca vacío. En «Cargar resultados» seleccionar `examples/datos-ensayo.json` para probar filtros, gráficos y validación con valores inventados del run sintético. Sus códigos son de ensayo; no corresponden a ubicaciones reales. El mapa permanece pendiente hasta disponer de cartografía que corresponda a los datos.

También admite un CSV de seis columnas, en el orden exacto de entrega. Un CSV aislado se muestra como procedencia sin verificar y no incluye métricas. El JSON del exportador añade run, hashes, marca de ensayo, métricas y cartografía opcional. Al reemplazar resultados se elimina la cartografía anterior para evitar uniones accidentales entre archivos.

## Conectar el run real

```bash
# Descargar una versión de geometrías abiertas antes de la presentación.
uv run --frozen python -m tecton geography

# Sustituir RUN_ID por el run completo que se quiere visualizar.
uv run --frozen python -m tecton dashboard \
  --run-id RUN_ID \
  --geojson data/geography/municipios.geojson

python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory delivery/RUN_ID/geovisor
```

Si ya hay champion real, omitir `--run-id`. Una exportación existente no se sobrescribe; para otra versión usar `--out delivery/visor-v2`. `tecton bundle` incluye automáticamente el geovisor cuando existe `dashboard/dist`; incorpora la cartografía convencional `data/geography/municipios.geojson` si está disponible. Se mantiene `dashboard.html` como respaldo sin WebGL.

El exportador verifica que el run terminó con QA, comprueba los hashes de sus artefactos, confirma que los CSV originales no cambiaron y vuelve a validar llaves, orden y dominios de predicción. Un run sintético exige `--demo`. La interfaz consume un snapshot: entrenar otro modelo no modifica un visor ya exportado.

## Cartografía y descarga manual

Fuente primaria: [DANE, MGN2024, capa Municipio](https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2024/Serv_CapasMGN_2024/FeatureServer/317). El campo `mpio_cdpmp` concatena departamento y municipio, con cinco caracteres. La descarga paginada solicita EPSG:4326 y una simplificación máxima de 0,003 grados para visualización. Conserva URL, fecha, versión y SHA256. No usar esa geometría simplificada para calcular áreas o covariables.

La descarga directa fue bloqueada con HTTP 403 en el entorno de preparación; no está incluida ni se afirma una cobertura nacional verificada. Ejecutar el comando en la PC. Si la fuente falla también allí, obtener GeoJSON del servicio oficial o exportarlo con QGIS y cargarlo manualmente. Los requisitos son:

- `FeatureCollection` con geometría `Polygon` o `MultiPolygon` en longitud/latitud WGS84.
- Propiedad `DIVIPOLA` o `mpio_cdpmp` como texto de cinco dígitos.
- Un polígono por código; disolver geometrías multipartes en `MultiPolygon` si hay repetidos.
- Nombres opcionales: `mpio_cnmbr` y `dpto_cnmbr`, o `municipio` y `departamento`.

La capa de una versión administrativa puede contener códigos distintos de los 1.102 del reto. No asumir igualdad. El visor cuenta las coincidencias y mantiene municipios sin geometría en los KPIs, gráficos y tabla. No hay uniones aproximadas por nombre. Cargar cartografía manual exige declarar su fuente en la documentación de la entrega; si solo enriquece el visor, aclarar que no alimentó el entrenamiento.

## Vistas y definición de cifras

1. **Explorar territorio:** mes, departamento y búsqueda afectan mapa, ranking, KPIs y tabla. La variable del mapa ordena también el ranking y la tabla. La búsqueda por nombre requiere nombres en la cartografía. El detalle del municipio muestra todos sus meses y está rotulado como tal.
2. **Validación del modelo:** AUC, RMSE log, Winkler log, cobertura, folds y stress del `metrics.json` del run. Las métricas no cambian con filtros geográficos y no corresponden al conjunto de prueba. Un CSV sin etiquetas no permite calcularlas.
3. **Fuentes y metodología:** interpretación del target, límites de cuantiles, procedencia, hashes y fechas.

| Cifra | Definición / límite |
|---|---|
| Municipios en la selección | Filas del mes que pasan departamento y búsqueda; una por código |
| Probabilidad ≥ umbral | Conteo entre esas mismas filas; umbral exploratorio ajustable, no alerta oficial |
| Suma de personas estimadas | Suma de puntos del modelo, no conteo observado ni media insesgada garantizada |
| Amplitud | q90 − q10, en personas; no es riesgo ni confianza del modelo |
| Color de probabilidad | Seis tramos con dominio fijo 0–1, comparable entre meses y filtros |
| Color de magnitud/amplitud | Seis tramos en log(1 + valor), dominio máximo del archivo completo; leyenda en personas |
| Intervalo q10–q90 | Nominal 80% por municipio-mes; no se suman extremos para producir un intervalo nacional |

La guía llama a la columna `personas_desplazadas`, pero su diccionario la define como personas reportadas como **afectadas**. La interfaz lo explica sin cambiar nombres del CSV. Evento también puede significar viviendas afectadas con cero personas.

El fondo del mapa es local, sin tiles, geocodificación ni tokens Mapbox. Esto reduce dependencias de red; se ven polígonos y tooltips sin nombres de carreteras o ciudades del mapa base. El mapa necesita WebGL. Tabla y gráficos siguen operativos cuando el mapa no puede iniciarse. La descarga de selección conserva las seis columnas, pero no sustituye el CSV completo de 42.978 filas.

## Colab y Claude Code

El notebook sigue ejecutando el mismo motor Python y muestra el HTML de respaldo. Para explorar un entrenamiento hecho en Colab, descargar su `predicciones.csv` y cargarlo en el visor local. No se afirma que Kepler esté incrustado ni que WebGL se haya probado dentro de Colab. El frontend se desarrolla en la PC y consume el mismo contrato, sin duplicar el modelo.

La regla `.claude/rules/dashboard.md` fija invariantes de la interfaz. El prompt para Claude está en `docs/PROMPT_DASHBOARD.md`.

## Desarrollo y comprobaciones

```bash
cd dashboard
npm ci
npm test
npm run build
# Solo en la PC, para desarrollar:
npm run dev
```

Node >=20.19, preferiblemente la misma familia Node 24 usada para compilar. Tras reconstruir se puede servir `dist` con Python. El ZIP incluye esa compilación, sin `node_modules`.

Antes de presentar, probar en Chromium/Firefox de la PC: cargar JSON y GeoJSON; cambiar mes y variable; confirmar nombres y código 05001; seleccionar un municipio desde el mapa; filtrar por departamento; comparar conteo de filas/mapa y municipios sin geometría; revisar escala fija; descargar selección; abrir validación; comprobar vista móvil y navegación por teclado. Esta prueba visual de WebGL queda pendiente porque el entorno de preparación no ofrece el navegador requerido para QA local.

El mapa se carga de forma diferida al haber geometría. La compilación Kepler es grande: aproximadamente 12 MB de JavaScript sin compresión, además de un recurso WASM de sus dependencias. No representa tamaño de los datos reales ni una medición de fluidez en la PC. Si una optimización se justifica después de validar, congelar primero esta versión y medir antes/después.

## Respaldo de las decisiones

- Guía adjunta, páginas 3–4: entregables, contrato CSV, métricas y confidencialidad; página 6: significado del target.
- [Kepler.gl: acciones](https://docs.kepler.gl/docs/api-reference/actions/actions): datasets, campos y `addDataToMap`.
- [Kepler.gl: estilos de mapa](https://github.com/keplergl/kepler.gl/blob/master/docs/api-reference/advanced-usages/custom-map-styles.md): estilos propios y reemplazo de estilos predeterminados.
- [DANE: metadatos de la capa municipal](https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2024/Serv_CapasMGN_2024/FeatureServer/317?f=pjson): campos y geometrías oficiales.
- Versiones efectivamente instaladas y lockfile del proyecto, no versiones supuestas del repositorio master. Se eligió Kepler 3.2.6 estable frente a la etiqueta latest que apuntaba a una versión alpha.
