# Comparativa OCR clásico vs VLM con RT-DETRv2 como detector base

Fuentes: `RT-DETR_PaddleOCR.ipynb`  y `RT-DETR_VLM.ipynb` , más `utils_matriculas.py`.
En ambos casos el detector de matrículas es RT-DETRv2 tamaño `s` y el de vehículos/personas RT-DETRv2 tamaño `l`, con ByteTrack. Todas las ejecuciones se hicieron en la misma máquina  y sobre el mismo dataset (`plates/raw-dataset`, 200 imágenes, 407 placas).

## 1. Configuración de cada experimento

| Aspecto | RT-DETR + PaddleOCR | RT-DETR + VLM |
|---|---|---|
| Lector de texto | PaddleOCR  | PaddleOCR-VL  |
| Confianza de la lectura | Media de la confianza del OCR | Media de probabilidad de los tokens |
| Umbral del detector en la evaluación | 0.35 | 0.25 |
| Preprocesado del detector | `preprocess` 640x640 (deforma la imagen), tamaños `(w, h)` | `letterbox` (mantiene proporción) |
| Evaluador | `evaluar_dataset` de utils | Bucle propio del cuaderno con `leer_cronometrado` |
| Corrección por posición | No | Sí  |
| Vídeo: `OCR_CADA` / `OCR_MAX_INTENTOS` | 5 / 6 | 5 / 6 |

## 2. Resultados en el dataset anotado

### 2.1 Detección

| Métrica | PaddleOCR | VLM |
|---|---|---|
| Detecciones RT-DETR | 567 | 613 |
| Placas anotadas encontradas | 188 / 407 | 182 / 407 |
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

Diferencias, en puntos porcentuales:

| Escenario | Exact match | Por carácter |
|---|---|---|
| Techo (mismo conjunto de 210 recortes) | +8,1 (+10,9 con corrección) | +4,4 (+5,0 con corrección) |
| Detecciones (conjuntos distintos) | +12,2 (+16,1 con corrección) | +4,7 (+5,7 con corrección) |

### 2.3 Tiempo por lectura

| Escenario | PaddleOCR | VLM (media / mediana) | Factor (media) |
|---|---|---|---|
| Detecciones | 403,20 ms | 1029 ms / 965 ms | ~2,6x |
| Recorte anotado (techo) | 601,02 ms | 1086 ms / 966 ms | ~1,8x |

## 3. Conclusiones

- **Precisión**: el VLM lee mejor. En el techo gana 8,1 puntos de exact match llegando a  10,9 con corrección y 4,4 por carácter.
- **Coste**: el VLM es entre 1,8 y 2,6 veces más lento de media, aunque con una ventaja de hardware debido a que el VLM es ejecutado con GPU, mientras que el OCR es ejecutado a través de CPU.
- **Detector**: con umbral 0.35 y `preprocess`, RT-DETR + PaddleOCR produjo 278 falsos positivos geométricos sobre 567 detecciones. Con 0.25 y `letterbox`, RT-DETR + VLM produjo 613 detecciones y encontró 182 placas, 6 menos que con el otro preprocesado.