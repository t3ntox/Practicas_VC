# Comparativa OCR clásico vs VLM con RT-DETRv2 como detector base

Fuentes: `RT-DETR_PaddleOCR.ipynb` (OCR clásico: PaddleOCR, modelos PP-OCRv6 det/rec) y `RT-DETR_VLM.ipynb` (VLM: `PaddlePaddle/PaddleOCR-VL`).
En ambos casos el detector de matrículas es RT-DETRv2 tamaño `s` y el de vehículos/personas RT-DETRv2 tamaño `l`, con ByteTrack.

## 1. Configuración de cada experimento

| Aspecto | RT-DETR + PaddleOCR | RT-DETR + VLM |
|---|---|---|
| Lector de texto | PaddleOCR (det + rec) | PaddleOCR-VL (bfloat16 en CUDA) |
| Confianza de la lectura | Confianza media del OCR | Media de probabilidad de los tokens generados |
| Umbral de confianza del detector (evaluación) | 0.35 | 0.25 |
| Preprocesado en evaluación | `preprocess` 640x640 | `letterbox` (mantiene proporción) |
| Carpeta del dataset | `plates/raw-dataset` | `datos` |
| Imágenes evaluadas | 200 | 200 |
| Corrección por posición | No | Sí (`corregir`) |
| Vídeo: `OCR_CADA` / `OCR_MAX_INTENTOS` | 5 / 6 | 5 / 6 |
| Carpeta de resultados de vídeo | `RT-DETR_PaddleOCR_result` | `RT-DETR_VLM_result` |

## 2. Resultados en el dataset anotado (200 imágenes)

### 2.1 Detección

| Métrica | PaddleOCR | VLM |
|---|---|---|
| Placas anotadas | 407 (todas con texto) | 411 |
| Detecciones RT-DETR | 567 | 516 |
| Placas anotadas encontradas | 188 / 407 | 198 / 411 |
| Falsos positivos geométricos (IoU < 0.5) | 278 | no reportado |
| Detecciones sobre placas sin texto | 101 | no reportado |

### 2.2 Exactitud de lectura

| Escenario | Métrica | PaddleOCR | VLM | VLM + corrección |
|---|---|---|---|---|
| Sobre detecciones de RT-DETR | n | 188 | 198 | 198 |
| | Exact match | 0.702 | 0.803 | 0.838 |
| | Acierto por carácter | 0.881 | 0.931 | 0.934 |
| Sobre recorte anotado (techo del OCR) | n | 210 | 208 | 208 |
| | Exact match | 0.648 | 0.774 | 0.808 |
| | Acierto por carácter | 0.791 | 0.917 | 0.922 |

Diferencias (VLM frente a PaddleOCR):

| Escenario | Exact match | Por carácter |
|---|---|---|
| Detecciones | +10,1 pp (+13,6 pp con corrección) | +5,0 pp |
| Techo (recorte anotado) | +12,6 pp (+16,0 pp con corrección) | +12,6 pp |

### 2.3 Tiempo por lectura

| Escenario | PaddleOCR | VLM (media / mediana) | Factor (media) |
|---|---|---|---|
| Detecciones | 403 ms | 1045 ms / 568 ms | ~2,6x |
| Recorte anotado | 601 ms | 1032 ms / 566 ms | ~1,7x |

## 3. Vídeo (`C0142.MP4`)

Los cuadernos de RT-DETR no muestran la salida de `procesar_video_rtdetr` (variables `plates_seen` y `lecturas` sin imprimir), así que **no hay cifras de vídeo para esta base**. Ambas configuraciones usan los mismos parámetros de OCR (cada 5 fotogramas, hasta 6 intentos, confianza mínima 0.5), por lo que una ejecución posterior sería directamente comparable.

## 4. Conclusiones

- **Precisión**: el VLM lee mejor en todos los escenarios, con una ventaja de entre 10 y 13 puntos en exact match. La corrección por posición añade unos 3 a 4 puntos más.
- **Robustez**: la mayor distancia aparece en el techo (recorte anotado), donde el VLM gana 12,6 puntos también por carácter. Indica que el VLM tolera mejor recortes difíciles (placas pequeñas, dos líneas, desenfoque), mientras PaddleOCR cae más.
- **Coste**: el VLM es entre 1,7 y 2,6 veces más lento de media. La mediana (~567 ms) es mucho menor que la media (~1040 ms), lo que sugiere una cola de lecturas muy lentas (probablemente generación de más tokens) y no un coste uniforme.
- **Detector**: con umbral 0.35 RT-DETR produjo 567 detecciones, con 278 falsos positivos geométricos. Encontró 188 de 407 placas, el mismo número que YOLO, pero con muchos más falsos positivos.

## 5. Advertencias para interpretar la comparación

1. **No es una comparación controlada**. Difieren el umbral de confianza del detector (0.35 vs 0.25), el preprocesado, la carpeta de datos y el número de placas anotadas (407 vs 411). Los `n` no coinciden (188 vs 198 y 210 vs 208).
2. **Hardware distinto**: las rutas indican que PaddleOCR se ejecutó en una máquina (`vidal`) y el VLM en otra (`enriq`). Los tiempos no son estrictamente comparables.
3. Que el exact match en detecciones supere al del techo (p. ej. 0.702 vs 0.648 en PaddleOCR) es un efecto de selección: solo se evalúan detecciones que casan con una anotación (IoU ≥ 0.5) y con texto válido.
4. Para una comparación limpia conviene fijar el mismo umbral, dataset y máquina en ambos cuadernos.
