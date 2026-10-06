# Contrato conocido del reto

Fuente: guía PNUD adjunta a esta conversación, pp. 1–8. Resumen verificado el 6 de octubre de 2026. Los T&C separados y el CSV de formato no estaban disponibles al crear el scaffold; revisar los originales cuando se entreguen.

- Evento: martes 6 de octubre de 2026, Bogotá. Ingreso antes de las 8:00; apertura de datos 10:00. Cierre confirmado por el equipo: 14:45 hora Colombia (la diapositiva decía 15:00 y el notebook base 14:00).
- 1102 municipios por mes. Entrenamiento 2018-01–2022-09, 62814 filas. Prueba equipos 2022-10–2024-04, 20938. Oculta 2024-05–2025-12, 22040.
- Target riesgo: tiene_evento=1 si hubo evento climático con personas o viviendas afectadas. Puede ser 1 con cero personas.
- personas_desplazadas agrega personas reportadas como afectadas según el diccionario. No inferir que esta definición demuestra desplazamiento individual observado.
- CSV único: 42978 filas, DIVIPOLA, fecha, prob_evento, personas_desplazadas_estimadas, personas_desplazadas_q10, personas_desplazadas_q90.
- AUC 50%; RMSE log1p sobre todas las filas 20%; Winkler log1p alfa0.2 5%; replicabilidad 5%; panel de finalistas 20%.
- Leaderboard público y clasificación oculta usan períodos distintos. No seleccionar por una subida aislada pública.
- Máximo una entrega cada 30 minutos; cuenta la última válida antes del cierre.
- Entregar CSV, código/notebook, geovisor o dashboard y tabla de replicabilidad. Finalistas: documento de máximo dos páginas.
- IA generativa permitida como apoyo del modelo/código con declaración de herramienta, versión y uso. No compartir/publicar datos del reto.
- Fuentes abiertas adicionales permitidas con declaración, excepto reportes de emergencias de períodos de prueba o reconstrucción de targets.
- Ingresos es per cápita. Inversión riesgo es total y su valor 2020 se repite hasta 2025. ONI entregado es una señal nacional centrada en el mes.

Pendientes de aclaración: autorización específica de covariables meteorológicas observadas durante el mes de prueba; reglas de plataforma local vs notebook Colab; tratamiento de nulos; normalización exacta de RMSE/Winkler de T&C 6.3. El scaffold no supone respuestas.

## Actualización del 6 de octubre (diapositivas, organizador y notebook base)

- Fuentes externas: solo delimitadas a Colombia; las globales se permiten pero el organizador las desaconseja. Nada anterior a 2018. Nada observado en los meses de prueba. Decisión: ningún dato externo en el modelo (ver docs/fuentes_datos.csv). Proxies globales solo en la tabla de replicabilidad.
- Entrenamiento y validación se ejecutan en Google Colab. Entregable de código: enlace de lectura a la copia de Colab del equipo.
- Datos: hackathon_datos.zip por enlace de Drive (gdown) con los tres CSV y diccionario_datos.csv. El enlace no se versiona: el repo GitHub es público.
- Nota automática (máx. 75), notebook base T&C 6.3: 100*(0.5*max(0,(AUC-0.5)/0.5) + 0.2*clip(1-RMSE/RMSE_nulo) + 0.05*clip(1-Winkler/Winkler_nulo)); nulo = prob 0.5 y cero personas con intervalo [0,0]. Implementado como puntos_75.
- Descalifica: reconstruir el oculto, predicciones fijas o a mano, compartir datos fuera del equipo, plagio sin citar, entregar tarde o por otro canal. Declarar IA en modelo y código (herramienta, versión, uso) y datos externos; el geovisor no requiere declaración.
- Intervalo q10-q90 debe cubrir cerca del 80%.
