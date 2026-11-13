# Comparativa OCR clásico vs VLM con RT-DETRv2 como detector base

Para esta comparativa se parte del detector de matrículas RT-DETR, el modelo `s`. El OCR clásico a analizar es PaddleOCR y el VLM es PaddleOCR-VL, cargado desde Hugging Face.
Los resultados son producto del procesamiento de las mismas 200 imágenes del dataset (411 placas anotadas, todas con texto).

## 1. Configuración de cada experimento

| Aspecto | RT-DETR + PaddleOCR | RT-DETR + Hugging Face |
|---|---|---|
| Lector de texto | PaddleOCR | PaddleOCR-VL |
| Salida de la evaluación | `ocr_eval/` | `vlm_eval/` |
| Confianza de la lectura | Media de la confianza del OCR | Media de probabilidad de los tokens |
| Umbral del detector en la evaluación | 0.35 | 0.35 |
| Preprocesado de la imagen | `preprocess` | `preprocess` |
| Corrección por posición | Sí | Sí |
| Vídeo: `OCR_CADA` / `OCR_MAX_INTENTOS` | 5 / 6 | 5 / 6 |

Ambos experimentos usan el mismo detector con la misma configuración, por lo que producen exactamente las mismas detecciones y la comparación sobre detecciones es directa.

## 2. Resultados en el dataset anotado

### 2.1 Detección (idéntica en ambos)

| Métrica | Valor |
|---|---|
| Imágenes evaluadas | 200 |
| Placas anotadas | 411 |
| Detecciones RT-DETR | 399 |
| Placas anotadas encontradas | 199 / 411 |
| Falsos positivos geométricos (IoU < 0.5) | 136 |
| Detecciones sobre placas sin texto | 64 |

### 2.2 Exactitud de lectura

| Escenario | Métrica | PaddleOCR | VLM | PaddleOCR + corrección | VLM + corrección |
|---|---|---|---|---|---|
| Sobre detecciones de RT-DETR | n | 199 | 199 | 199 | 199 |
| | Exact match | 0.714 | 0.814 | 0.754 | 0.854 |
| | Acierto por carácter | 0.889 | 0.932 | 0.900 | 0.942 |
| Sobre recorte anotado (techo) | n | 208 | 208 | 208 | 208 |
| | Exact match | 0.716 | 0.788 | 0.736 | 0.822 |
| | Acierto por carácter | 0.881 | 0.924 | 0.885 | 0.932 |

Diferencias del VLM frente a PaddleOCR, en puntos porcentuales (se compara lectura cruda con cruda y corregida con corregida):

| Escenario | Exact match (crudo) | Exact match (con corrección) | Por carácter (crudo) | Por carácter (con corrección) |
|---|---|---|---|---|
| Detecciones | +10,0 | +10,0 | +4,3 | +4,2 |
| Techo | +7,2 | +8,6 | +4,3 | +4,7 |

Mejora que aporta la corrección por posición sobre cada lector (exact match, en puntos):

| Escenario | PaddleOCR | VLM |
|---|---|---|
| Detecciones | +4,0 | +4,0 |
| Techo | +2,0 | +3,4 |

### 2.3 Tiempo por lectura

| Escenario | PaddleOCR (media / mediana) | VLM (media / mediana) | Factor (media) | Factor (mediana) |
|---|---|---|---|---|
| Detecciones | 416 ms / 330 ms | 777 ms / 519 ms | ~1,9x | ~1,6x |
| Recorte anotado (techo) | 417 ms / 333 ms | 871 ms / 517 ms | ~2,1x | ~1,6x |

La mediana del VLM (~518 ms) es prácticamente la misma que con YOLO, mientras que la media se dispara (777-871 ms). Esto indica lecturas puntuales muy lentas que inflan la media, no una lectura típica más costosa, por lo que la mediana es la medida más fiable.

## 3. Conclusiones

- **Precisión**: el VLM lee mejor. Sobre las mismas 199 detecciones gana 10,0 puntos de exact match (también 10,0 con corrección) y 4,3 por carácter. En el techo la ventaja es de 7,2 puntos de exact match (8,6 con corrección) y 4,3 por carácter.
- **Coste**: el VLM es entre ~1,9x y ~2,1x más lento de media y ~1,6x de mediana, con la ventaja de hardware de que el VLM se ejecuta en GPU y PaddleOCR en CPU.
- **Corrección por posición**: aporta entre 2,0 y 4,0 puntos de exact match según el escenario, y beneficia más al VLM en el techo (+3,4 frente a +2,0).
- **Detector**: RT-DETR genera 399 detecciones, de las cuales 136 son falsos positivos geométricos y 64 caen sobre placas sin texto, y encuentra 199 de las 411 placas. Frente a YOLO (376 detecciones, 55 falsos positivos y 210 placas encontradas), RT-DETR es menos preciso en detección.
