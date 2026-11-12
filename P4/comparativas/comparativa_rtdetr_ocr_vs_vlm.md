# Comparativa OCR clásico vs VLM con RT-DETRv2 como detector base

Fuentes: `RT-DETR_PaddleOCR.ipynb` (OCR clásico: PaddleOCR) y `RT-DETR_VLM.ipynb` (VLM: `PaddlePaddle/PaddleOCR-VL`), más `utils_matriculas.py`.
En ambos casos el detector de matrículas es RT-DETRv2 tamaño `s` y el de vehículos/personas RT-DETRv2 tamaño `l`, con ByteTrack. Todas las ejecuciones se hicieron en la misma máquina (rutas de `vidal`) y sobre el mismo dataset (`plates/raw-dataset`, 200 imágenes, 407 placas anotadas con texto).

## 1. Configuración de cada experimento

| Aspecto | RT-DETR + PaddleOCR | RT-DETR + VLM |
|---|---|---|
| Lector de texto | PaddleOCR (en `utils`, `device='cpu'`) | PaddleOCR-VL (CUDA, bfloat16) |
| Confianza de la lectura | Media de la confianza del OCR | Media de probabilidad de los tokens |
| Umbral del detector en la evaluación | 0.35 | 0.25 |
| Preprocesado del detector | `preprocess` 640x640 (deforma la imagen), tamaños `(w, h)` | `letterbox` (mantiene proporción) |
| Evaluador | `evaluar_dataset` de utils | Bucle propio del cuaderno con `leer_cronometrado` |
| Corrección por posición | No | Sí (`corregir`) |
| Vídeo: `OCR_CADA` / `OCR_MAX_INTENTOS` | 5 / 6 | 5 / 6 |

## 2. Resultados en el dataset anotado

### 2.1 Detección

| Métrica | PaddleOCR | VLM |
|---|---|---|
| Detecciones RT-DETR | 567 | 613 |
| Placas anotadas encontradas (con texto) | 188 / 407 | 182 / 407 |
| Falsos positivos geométricos (IoU < 0.5) | 278 | no reportado |
| Detecciones sobre placas sin texto | 101 | no reportado |

### 2.2 Exactitud de lectura

| Escenario | Métrica | PaddleOCR | VLM | VLM + corrección |
|---|---|---|---|---|
| Sobre detecciones de RT-DETR | n | 188 | 182 | 182 |
| | Exact match | 0.702 | 0.824 | 0.863 |
| | Acierto por carácter | 0.881 | 0.928 | 0.938 |
| Sobre recorte anotado (techo) | n | 210 | 210 | 210 |
| | Exact match | 0.648 | 0.729 | 0.757 |
| | Acierto por carácter | 0.791 | 0.835 | 0.841 |

Diferencias (VLM frente a PaddleOCR), en puntos porcentuales:

| Escenario | Exact match | Por carácter |
|---|---|---|
| Techo (mismo conjunto de 210 recortes) | +8,1 (+10,9 con corrección) | +4,4 (+5,0 con corrección) |
| Detecciones (conjuntos distintos) | +12,2 (+16,1 con corrección) | +4,7 (+5,7 con corrección) |

La fila del **techo** es la comparación limpia: son exactamente los mismos recortes y no depende del detector. La de detecciones mezcla umbrales y subconjuntos distintos, así que sobrestima la ventaja del VLM.

### 2.3 Tiempo por lectura

| Escenario | PaddleOCR | VLM (media / mediana) | Factor (media) |
|---|---|---|---|
| Detecciones | 403,20 ms | 1029 ms / 965 ms | ~2,6x |
| Recorte anotado (techo) | 601,02 ms | 1086 ms / 966 ms | ~1,8x |

## 3. Conclusiones

- **Precisión**: el VLM lee mejor. En el techo gana 8,1 puntos de exact match (10,9 con corrección) y 4,4 por carácter. La corrección por posición añade unos 3 puntos.
- **Coste**: el VLM es entre 1,8 y 2,6 veces más lento de media, aunque con una ventaja de hardware (GPU frente a la CPU de PaddleOCR).
- **Detector**: con umbral 0.35 y `preprocess`, RT-DETR + PaddleOCR produjo 278 falsos positivos geométricos sobre 567 detecciones. Con 0.25 y `letterbox`, RT-DETR + VLM produjo 613 detecciones y encontró 182 placas, 6 menos que con el otro preprocesado.

## 4. Advertencias

1. **Los detectores no están en igualdad**: umbral (0.35 vs 0.25) y preprocesado (`preprocess` vs `letterbox`) cambian entre cuadernos, por eso `n` difiere (188 vs 182). Para comparar OCR, usa el techo.
2. **Dispositivo**: `utils_matriculas.py` inicializa PaddleOCR con `device='cpu'`, mientras el VLM corre en CUDA. Los tiempos no son a igualdad de hardware.
3. **Cronometraje no equivalente**: en PaddleOCR (`evaluar_dataset`) el tiempo incluye el recorte y se promedia sobre todas las detecciones, también los falsos positivos. En el VLM (`leer_cronometrado`) solo se mide la lectura y solo en detecciones emparejadas.
4. **Variación entre ejecuciones**: el mismo trabajo (techo del VLM, 210 recortes) tardó ~765 ms en `YOLO_VLM` y 1086 ms en este cuaderno. Diferencias de ese orden no se pueden atribuir al modelo.
5. **Posible causa del retraso**: en este cuaderno, la celda de evaluación carga una segunda copia del VLM (`vlm`, `procesador`) que no se usa (`leer_placa` usa `ocr_global`). Duplica memoria de GPU; es una hipótesis, no está medido.
6. Lo idéntico por construcción (utils): recorte (`_preparar_recorte`), normalización, métricas (`lev`, exact match, por carácter) y vídeo (`procesar_video_rtdetr`).
