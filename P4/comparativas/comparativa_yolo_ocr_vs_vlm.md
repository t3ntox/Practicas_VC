# Comparativa OCR clásico vs VLM con YOLO como detector base

Para esta comparativa se parte del detector de matrículas YOLO, el modelo `s`. El OCR clásico a analizar es PaddleOCR y el VLM es PaddleOCR-VL, cargado desde Hugging Face.
Los resultados son producto del procesamiento de las mismas 200 imágenes del dataset (411 placas anotadas, todas con texto).

## 1. Configuración

| Aspecto | YOLO + PaddleOCR | YOLO + VLM |
|---|---|---|
| Lector de texto | PaddleOCR | PaddleOCR-VL |
| Salida de la evaluación | `ocr_eval/` | `vlm_eval/` |
| Corrección por posición | Sí | Sí |
| Vídeo: `OCR_CADA` / `OCR_MAX_INTENTOS` | 5 / 6 | 10 / 3 |
| Vídeo: resto de parámetros | `CONF_PLACA` 0.4, `MIN_FRAMES_PLACA` 5, `MIN_FRAMES` 10 | Iguales |

## 2. Resultados en el dataset anotado

### 2.1 Detección (idéntica en ambos)

| Métrica | Valor |
|---|---|
| Imágenes evaluadas | 200 |
| Placas anotadas | 411 |
| Detecciones YOLO | 376 |
| Placas anotadas encontradas | 210 / 411 |
| Falsos positivos geométricos (IoU < 0.5) | 55 |
| Detecciones sobre placas sin texto | 111 |

El detector es el mismo en ambos experimentos, por lo que estas cifras no dependen del lector.

### 2.2 Exactitud de lectura

| Escenario | Métrica | PaddleOCR | VLM | PaddleOCR + corrección | VLM + corrección |
|---|---|---|---|---|---|
| Sobre detecciones de YOLO | n | 210 | 210 | 210 | 210 |
| | Exact match | 0.714 | 0.786 | 0.752 | 0.824 |
| | Acierto por carácter | 0.888 | 0.919 | 0.898 | 0.931 |
| Sobre recorte anotado (techo) | n | 208 | 208 | 208 | 208 |
| | Exact match | 0.716 | 0.788 | 0.736 | 0.822 |
| | Acierto por carácter | 0.881 | 0.924 | 0.885 | 0.932 |

Diferencias del VLM frente a PaddleOCR, en puntos porcentuales (se compara lectura cruda con cruda y corregida con corregida):

| Escenario | Exact match (crudo) | Exact match (con corrección) | Por carácter (crudo) | Por carácter (con corrección) |
|---|---|---|---|---|
| Detecciones | +7,2 | +7,2 | +3,1 | +3,3 |
| Techo | +7,2 | +8,6 | +4,3 | +4,7 |

Mejora que aporta la corrección por posición sobre cada lector (exact match, en puntos):

| Escenario | PaddleOCR | VLM |
|---|---|---|
| Detecciones | +3,8 | +3,8 |
| Techo | +2,0 | +3,4 |

### 2.3 Tiempo por lectura

| Medición | PaddleOCR (media / mediana) | VLM (media / mediana) | Factor (media) | Factor (mediana) |
|---|---|---|---|---|
| Detecciones | 415 ms / 333 ms | 540 ms / 514 ms | ~1,3x | ~1,5x |
| Recorte anotado (techo) | 425 ms / 338 ms | 538 ms / 512 ms | ~1,3x | ~1,5x |

## 3. Cruce detector × OCR (todos los cuadernos)

| Combinación | Detecciones | Placas halladas | Falsos positivos (IoU < 0.5) | Exact match (detecciones) | Con corrección | Por carácter | Tiempo medio |
|---|---|---|---|---|---|---|---|
| YOLO + PaddleOCR | 376 | 210 | 55 | 0.714 | 0.752 | 0.888 | 415 ms |
| YOLO + VLM | 376 | 210 | 55 | 0.786 | 0.824 | 0.919 | 540 ms |
| RT-DETR + PaddleOCR (conf 0.35) | 399 | 199 | 136 | 0.714 | 0.754 | 0.889 | 416 ms |
| RT-DETR + VLM (conf 0.35) | 399 | 199 | 136 | 0.814 | 0.854 | 0.932 | 777 ms |

- El techo (208 recortes anotados) es idéntico con YOLO y con RT-DETR para cada lector, como cabe esperar al no intervenir el detector.
- Con RT-DETR, PaddleOCR y VLM evalúan exactamente las mismas 399 detecciones (n=199), así que esa comparación es directa.
- El exact match sobre detecciones no permite declarar un detector ganador: los conjuntos evaluados difieren (n=199 con RT-DETR frente a n=210 con YOLO). RT-DETR + VLM sale algo por encima de YOLO + VLM (0.814 frente a 0.786), pero sobre menos placas, y con PaddleOCR el resultado es idéntico (0.714).
- La diferencia de detección sí es clara: YOLO encuentra más placas (210 frente a 199) con muchos menos falsos positivos (55 frente a 136) y menos detecciones totales (376 frente a 399).

## 4. Conclusiones

1. **El VLM lee mejor que PaddleOCR con el mismo detector**: +7,2 puntos de exact match tanto sobre las detecciones como sobre el techo, y +3,1 y +4,3 puntos por carácter respectivamente. Con corrección por posición, la ventaja se mantiene (+7,2 en detecciones) o crece (+8,6 en el techo).
2. **El coste del VLM es moderado con YOLO**: es ~1,3x más lento de media y ~1,5x de mediana (540 ms frente a 415 ms de media). Hay que tener en cuenta que el VLM se ejecuta en GPU y PaddleOCR en CPU.
3. **La corrección por posición aporta de 2,0 a 3,8 puntos de exact match**. Sobre detecciones aporta lo mismo a ambos lectores (+3,8); sobre el techo beneficia más al VLM (+3,4 frente a +2,0).
4. **YOLO es el detector más limpio**: 55 falsos positivos geométricos sobre 376 detecciones, además de encontrar más placas que RT-DETR.
