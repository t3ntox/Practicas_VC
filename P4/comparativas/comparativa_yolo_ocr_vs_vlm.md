# Comparativa OCR clásico vs VLM con YOLO como detector base

Fuentes: `YOLO_PaddleOCR.ipynb` (OCR clásico: PaddleOCR), `YOLO_VLM.ipynb` (VLM: `PaddlePaddle/PaddleOCR-VL`) y `utils_matriculas.py`.
Detector de matrículas: `weights/best-yolo.pt` en ambos. Tracking con ByteTrack, fusión de tracklets y validación por patrones de matrícula europeos. Todas las ejecuciones se hicieron en la misma máquina  sobre el mismo dataset (`plates/raw-dataset`, 200 imágenes, 407 placas anotadas con texto).

Esta es la comparación más limpia de las dos: mismo detector, mismo umbral (0.25, `imgsz` 960), mismas 367 detecciones y mismo evaluador (`evaluar_dataset` de utils).

## 1. Configuración

| Aspecto | YOLO + PaddleOCR | YOLO + VLM |
|---|---|---|
| Lector de texto | PaddleOCR | PaddleOCR-VL |
| Evaluación | `evaluar_dataset` | `evaluar_dataset` y `evaluar_dataset_vlm`  |
| Corrección por posición | No | Si  |
| Vídeo: `OCR_CADA` / `OCR_MAX_INTENTOS` | 5 / 6 | 10 / 3 |
| Vídeo: resto de parámetros | `CONF_PLACA` 0.4, `MIN_FRAMES_PLACA` 5, `MIN_FRAMES` 10 | Iguales |

## 2. Resultados en el dataset anotado

### 2.1 Detección (idéntica en ambos)

| Métrica | Valor |
|---|---|
| Detecciones YOLO | 367 |
| Placas anotadas encontradas | 188 / 407 |
| Falsos positivos geométricos (IoU < 0.5) | 108 |
| Detecciones sobre placas sin texto | 71 |

### 2.2 Exactitud de lectura

| Escenario | Métrica | PaddleOCR | VLM | VLM + corrección |
|---|---|---|---|---|
| Sobre detecciones de YOLO | n | 188 | 188 | 188 |
| | Exact match | 0.734 | 0.798 | 0.835 |
| | Acierto por carácter | 0.894 | 0.922 | 0.931 |
| Sobre recorte anotado (techo) | n | 210 | 210 | 210 |
| | Exact match | 0.648 | 0.729 | 0.757 |
| | Acierto por carácter | 0.791 | 0.835 | 0.841 |

Diferencias (VLM frente a PaddleOCR), en puntos porcentuales:

| Escenario | Exact match | Por carácter |
|---|---|---|
| Detecciones | +6,4 (+10,1 con corrección) | +2,8 (+3,7 con corrección) |
| Techo | +8,1 (+10,9 con corrección) | +4,4 (+5,0 con corrección) |

### 2.3 Tiempo por lectura

| Medición | PaddleOCR | VLM | Factor |
|---|---|---|---|
| Detecciones (`evaluar_dataset`, mismo método) | 501,84 ms | 650,17 ms | ~1,3x |
| Recorte anotado (`evaluar_dataset`, mismo método) | 629,43 ms | 764,55 ms | ~1,2x |
| Detecciones (`evaluar_dataset_vlm`, solo la lectura) | n/d | 749 ms (mediana 727) | |
| Recorte anotado (`evaluar_dataset_vlm`) | n/d | 767 ms (mediana 719) | |

Los dos evaluadores del VLM difieren  en alrededor de 100 ms sobre un trabajo equivalente (650 vs 749 ms), lo que da una idea de la variación entre ejecuciones.

## 3. Cruce detector × OCR (todos los cuadernos)

| Combinación | Detecciones | Placas halladas | Exact match (detecciones) | Por carácter |
|---|---|---|---|---|
| YOLO + PaddleOCR | 367 | 188 | 0.734 | 0.894 |
| YOLO + VLM | 367 | 188 | 0.798 | 0.922 |
| RT-DETR + PaddleOCR (conf 0.35) | 567 | 188 | 0.702 | 0.881 |
| RT-DETR + VLM (conf 0.25) | 613 | 182 | 0.824 | 0.928 |

- Con YOLO y RT-DETR el techo es idéntico por OCR.
- El detector influye poco en el exact match y de forma mixta: RT-DETR queda por debajo con PaddleOCR y algo por encima con el VLM. Los subconjuntos evaluados y los umbrales difieren, así que no conviene sacar un ganador por este dato.
- La diferencia de detección sí es clara: a umbral 0.25, RT-DETR genera 613 detecciones frente a 367 de YOLO, y encuentra menos placas.

## 4. Conclusiones

1. **El VLM lee mejor que PaddleOCR con el mismo detector**: +6,4 puntos de exact match en deteccione y +8,1 en el techo ambos siendo aun mayores cuando aplicamos las correciones.
2. **El coste es moderado en este cuaderno**: Este cuadernos es aproximadamente  de 1,2 a 1,3x más lento respecto a los otros .
3. **La corrección por posición aporta de 2,8 a 3,7 puntos de exact match** sobre el VLM.