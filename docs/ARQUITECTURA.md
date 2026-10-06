# Arquitectura del repositorio

Un solo repositorio. Tres frentes. El contrato de datos queda en el medio y ninguno de los tres lo redefine por su cuenta.

La raíz de este repo es el proyecto. Se trabaja desde aquí, no desde una carpeta `tecton-pnud/`.

| Frente | Quién | Encargo |
|---|---|---|
| Datos y modelo | Claude | Auditoría, features, entrenamiento, métricas, CSV de entrega y tabla de replicabilidad |
| Estética de la información | Cursor | Cómo se ven los datos, las gráficas y el mapa |
| Pulido de la interfaz | Laura | Jerarquía, espaciado, copy, estados, responsive y accesibilidad sobre lo que Cursor deje |

Los T&C del PNUD prevalecen sobre este documento. El resumen del reto está en `docs/RETO.md`. El estado de experimentos está en `docs/ESTADO.md`.

## Mapa

```text
TECTON/
├── CLAUDE.md                 memoria de Claude: datos y modelo
├── README.md                 instalación y flujo del harness
├── src/tecton/               motor Python. Dueño: Claude
│   ├── schema.py             contrato CSV de entrada y de entrega
│   ├── features.py           covariables e historial causal
│   ├── models.py             cuatro cabezas por motor
│   ├── metrics.py            AUC, RMSE log1p, Winkler, cobertura
│   ├── pipeline.py           folds, run, promoción
│   ├── dashboard.py          exporta el snapshot que consume la UI
│   └── artifacts.py          notebook, HTML de respaldo, replicabilidad, bundle
├── configs/                  protocolos de experimento. Dueño: Claude
├── tests/                    invariantes del motor. Dueño: Claude
├── notebooks/                plantilla Colab generada desde el mismo motor
├── data/raw/                 CSV oficiales. No se versionan
├── runs/                     evidencia de cada experimento. No se versiona
├── delivery/                 copia de entrega. No se versiona
├── state/                    champion y envíos. No se versiona
├── examples/datos-ensayo.json   fixture inventado para la UI, sin municipios reales
├── dashboard/src/            interfaz. Cursor implementa; Laura pule
│   ├── data.js               lectura del contrato. Cambio compartido
│   ├── App.jsx               composición de vistas
│   ├── KeplerMap.jsx         mapa
│   ├── kepler-config.js      estilo del mapa
│   ├── Trend.jsx             gráficas
│   └── styles.css            presentación visual
└── docs/
    ├── ARQUITECTURA.md       este archivo
    ├── RETO.md               contrato del hackathon
    ├── DASHBOARD.md          definición de cifras y vistas
    └── ESTADO.md             bitácora de experimentos
```

`dashboard/dist/` es salida de compilación y no se versiona. La fuente visual es `dashboard/src/`.

## Contrato entre frentes

Claude produce archivos. Cursor y Laura los presentan. La interfaz no reentrena ni recalcula la nota oficial.

Salida que se califica, 42.978 filas, en este orden:

`DIVIPOLA, fecha, prob_evento, personas_desplazadas_estimadas, personas_desplazadas_q10, personas_desplazadas_q90`

`DIVIPOLA` es texto de cinco dígitos. `fecha` es `AAAA-MM-01`. `prob_evento` está en 0–1. Las magnitudes son no negativas y `q10` ≤ `q90`.

Además, cada run completo deja:

| Archivo | Uso en la interfaz |
|---|---|
| `runs/RUN_ID/predicciones.csv` | Tabla, ranking, KPIs y mapa |
| `runs/RUN_ID/metrics.json` | Vista de validación: AUC, RMSE log, Winkler, cobertura, folds y estrés de 12 meses |
| `runs/RUN_ID/replicabilidad.csv` | Variables usadas y proxy global |
| `runs/RUN_ID/manifest.json` | Procedencia, hashes y marca de ensayo |
| `data/geography/municipios.geojson` | Polígonos WGS84 unidos por `DIVIPOLA` o `mpio_cdpmp`. No entra al entrenamiento |

Hasta que existan CSV oficiales, la UI se prueba con `examples/datos-ensayo.json`. Esos códigos no son ubicaciones reales.

Si Claude necesita exponer un campo nuevo, lo agrega en `src/tecton/dashboard.py`, lo documenta en `docs/DASHBOARD.md` y avisa el cambio de forma. Cursor no inventa columnas. Laura no renombra las seis columnas de entrega.

## Frente de Claude

Leer `CLAUDE.md`, `docs/RETO.md` y `docs/ESTADO.md`. Trabajar solo el motor.

Prioridad del score automático: riesgo 50%, magnitud 20%, intervalo 5%. La tabla en vivo usa `prueba_equipos`. La nota oficial usa `prueba_oculta`. No elegir un modelo por una subida aislada a la tabla pública.

El historial de tipos de evento (`n_eventos`, familias, viviendas y conteos por fenómeno) existe solo en entrenamiento y hoy no entra a `features.py`. Si se usa, tiene que ser anterior al mes predicho y quedar congelado en el corte. No usar reportes de emergencias de los periodos de prueba ni reconstruir targets.

Al cerrar un experimento: hipótesis, `run_id`, métricas agregadas, duración y decisión en `docs/ESTADO.md`. Sin filas, sin targets y sin OOF reales en el chat ni en git.

No editar `dashboard/src/App.jsx`, `KeplerMap.jsx`, `Trend.jsx`, `kepler-config.js` ni `styles.css`.

## Frente de Cursor

Implementar la estética de la información sobre el visor que ya existe. Tres vistas, definidas en `docs/DASHBOARD.md`:

1. Territorio: mes, departamento, búsqueda, mapa, ranking, KPIs y tabla comparten el mismo filtro.
2. Validación: métricas del run completo, no del filtro geográfico.
3. Fuentes: target, cuantiles, procedencia y hashes.

El mapa usa polígonos municipales, fondo local y dominio de color fijo. La probabilidad va de 0 a 1. La magnitud y la amplitud van en log1p, con leyenda en personas. Si WebGL no arranca, tabla y gráficas siguen visibles.

Puede cambiar composición, escala visual, gráfica y configuración del mapa en `dashboard/src/`. `data.js` se toca solo para presentar el contrato, no para alterar la definición de las cifras. Después de un cambio visual: `npm test` y `npm run build` dentro de `dashboard/`.

No modificar `src/tecton/features.py`, `models.py`, `pipeline.py`, `metrics.py` ni `schema.py`. No commitear `data/raw/`.

## Frente de Laura

Pulir y mejorar la interfaz cuando la vista de Cursor ya muestra datos, gráficas y mapa.

Su superficie es `dashboard/src/styles.css` y los detalles de presentación en `App.jsx`, `Trend.jsx` y `KeplerMap.jsx`: jerarquía, espaciado, tipografía, estados vacíos, mensajes de error, vista móvil, foco de teclado y redacción de la interfaz.

Las cifras conservan el significado de `docs/DASHBOARD.md`. Personas afectadas reportadas no se presentan como desplazamiento observado. El umbral de probabilidad no se llama alerta oficial. Los cuantiles no se suman para fabricar un intervalo nacional.

No cambia el protocolo de modelos, los folds ni el CSV de entrega.

## Orden

1. Claude deja un run con QA y un CSV válido, aunque sea la entrega provisional.
2. Cursor conecta ese resultado, o el ensayo, a gráficas y mapa con una lectura clara.
3. Laura recorre las tres vistas y cierra el pulido.

El HTML de `dashboard.html` que genera el harness sigue como respaldo sin WebGL. No se reemplaza por la interfaz de React.
