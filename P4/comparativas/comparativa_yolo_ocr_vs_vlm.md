# Comparativa OCR clásico vs VLM con YOLO como detector base

Fuentes: `YOLO_PaddleOCR.ipynb` (OCR clásico: PaddleOCR) y `YOLO_VLM.ipynb` (VLM: `PaddlePaddle/PaddleOCR-VL`).
Detector de matrículas: `weights/best-yolo.pt` en ambos. Tracking con ByteTrack (`track_buffer: 140`), fusión de tracklets y validación por patrones de matrícula europeos.

## 1. Configuración de cada experimento

| Aspecto | YOLO + PaddleOCR | YOLO + VLM |
|---|---|---|
| Lector de texto | PaddleOCR (`leer_placa_paddle`) | PaddleOCR-VL (`leer_placa_vlm`) |
| Evaluación en dataset | Ejecutada (200 imágenes) | **No ejecutada**: "Directorio plates\raw-dataset\images no encontrado" |
| Evaluación con corrección por posición | No existe | Definida, no ejecutada (mismo error de ruta) |
| Vídeo: `OCR_CADA` | 5 | 10 |
| Vídeo: `OCR_MAX_INTENTOS` | 6 | 3 |
| Resto de parámetros de vídeo | `CONF_PLACA` 0.4, `MIN_FRAMES_PLACA` 5, `MIN_FRAMES` 10 | Iguales |
| Carpeta de resultados | `YOLO_PaddleOCR_result` | `YOLO_VLM_result` |

## 2. Resultados en el dataset anotado

Solo existen datos de YOLO + PaddleOCR (conf 0.25, imgsz 960, 200 imágenes, 407 placas anotadas con texto):

| Métrica | YOLO + PaddleOCR |
|---|---|
| Detecciones YOLO | 367 |
| Placas anotadas encontradas | 188 / 407 |
| Falsos positivos geométricos (IoU < 0.5) | 108 |
| Detecciones sobre placas sin texto | 71 |
| Exact match sobre detecciones (n=188) | 0.734 |
| Acierto por carácter sobre detecciones | 0.894 |
| Exact match en techo (n=210) | 0.648 |
| Acierto por carácter en techo | 0.791 |
| Tiempo OCR por detección | 501,84 ms |
| Tiempo OCR por recorte anotado | 629,43 ms |

**YOLO + VLM no tiene métricas propias** porque su evaluación no llegó a ejecutarse. Como estimación orientativa, el techo del VLM no depende del detector. En `RT-DETR_VLM.ipynb` se obtuvo, sobre recortes anotados:

| Techo (recorte anotado) | PaddleOCR | VLM (medido en el cuaderno RT-DETR) |
|---|---|---|
| Exact match | 0.648 | 0.774 (0.808 con corrección) |
| Acierto por carácter | 0.791 | 0.917 (0.922 con corrección) |
| Tiempo medio | 629 ms (YOLO) / 601 ms (RT-DETR) | 1032 ms (mediana 566 ms) |

Esto sugiere que, con YOLO, el VLM mejoraría la lectura en unos 12 puntos de exact match en el techo, a cambio de ~1,7 veces más tiempo. Es una extrapolación, no un resultado medido con YOLO.

## 3. Vídeo (`C0142.MP4`)

| Métrica | YOLO + PaddleOCR | YOLO + VLM |
|---|---|---|
| IDs originales | 187 | 188 |
| IDs tras fusionar tracklets | 95 | 95 |
| Matrículas detectadas | 40 | 39 |
| Matrículas leídas con OCR | 23 (57,5 %) | 18 (46,2 %) |

- El seguimiento y la fusión se comportan casi igual (95 identidades finales en ambos).
- PaddleOCR lee 5 matrículas más. Esta diferencia **no puede atribuirse solo al modelo**: la configuración del VLM es más restrictiva (lectura cada 10 fotogramas y 3 intentos, frente a cada 5 y 6 de PaddleOCR), y por tanto tiene la mitad de oportunidades de leer cada vehículo.
- Además, el VLM estima su confianza a partir de probabilidades de tokens, lo que puede comportarse distinto frente al umbral `OCR_SCORE_OK = 0.5`. Es una hipótesis, no está medido.
- Detalle del cuaderno de PaddleOCR: la llamada a `procesar_video_yolo` pasa `etiqueta_ocr="VLM"`, probablemente un resto de copiar el otro cuaderno. Conviene cambiarlo a "PaddleOCR" para que las etiquetas del vídeo sean correctas.

## 4. Conclusiones

- Con YOLO como base, **la única comparación directa disponible es la de vídeo**, y no es equitativa por los parámetros distintos.
- YOLO es más selectivo que RT-DETR en la evaluación: 367 detecciones y 108 falsos positivos geométricos (conf 0.25), frente a 567 y 278 de RT-DETR (conf 0.35), localizando el mismo número de placas (188/407).
- YOLO + PaddleOCR rinde algo mejor que RT-DETR + PaddleOCR en detecciones (0.734 vs 0.702 de exact match, 0.894 vs 0.881 por carácter) y es más lento (502 vs 403 ms). El techo es idéntico (0.648 / 0.791), como cabe esperar al ser independiente del detector.

## 5. Siguientes pasos recomendados

1. Corregir la ruta `BASE` en `YOLO_VLM.ipynb` y ejecutar `evaluar_dataset` y `evaluar_dataset_vlm` para obtener métricas reales de YOLO + VLM.
2. Igualar en vídeo `OCR_CADA` y `OCR_MAX_INTENTOS` (p. ej. 5 y 6) para una comparación justa.
3. Ejecutar ambos en la misma máquina para que los tiempos sean comparables (las rutas indican equipos distintos: `vidal` y `enriq`).
