# Comparativa OCR clásico vs VLM con YOLO como detector base

Para esta comparativa se parte del detector de matrículas YOLO, el modelo `s`. El OCR clásico a analizar es PaddleOCR y el VLM es PaddleOCR-VL, cargado desde Hugging Face.
Los resultados son producto del procesamiento de las mismas 200 imágenes del dataset (411 placas anotadas, todas con texto).

## 1. Configuración de cada experimento

| Aspecto | YOLO + PaddleOCR | YOLO + VLM |
|---|---|---|
| Lector de texto | PaddleOCR | PaddleOCR-VL |
| Salida de la evaluación | `ocr_eval/` | `vlm_eval/` |
| Corrección por posición | Sí | Sí |
| Vídeo: `OCR_CADA` / `OCR_MAX_INTENTOS` | 5 / 6 | Iguales |
| Vídeo: resto de parámetros | `CONF_PLACA` 0.4, `MIN_FRAMES_PLACA` 5, `MIN_FRAMES` 10 | Iguales |

Ambos experimentos usan el mismo detector con la misma configuración, por lo que producen exactamente las mismas detecciones y la comparación sobre detecciones es directa. Los parámetros de vídeo `OCR_CADA` y `OCR_MAX_INTENTOS` solo afectan al procesamiento de vídeo y no intervienen en la evaluación sobre el dataset anotado.

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

El detector es el mismo en ambos experimentos, por lo que estas cifras no dependen del lector. YOLO genera 376 detecciones sobre las 200 imágenes y localiza 210 de las 411 placas anotadas, con 55 falsos positivos geométricos. La exactitud de lectura de la sección siguiente se calcula sobre esas 210 placas encontradas, de modo que ambos lectores se evalúan exactamente sobre el mismo conjunto de recortes.

### 2.2 Exactitud de lectura

| Escenario | Métrica | PaddleOCR | VLM | PaddleOCR + corrección | VLM + corrección |
|---|---|---|---|---|---|
| Sobre detecciones de YOLO | n | 210 | 210 | 210 | 210 |
| | Exact match | 0.714 | 0.786 | 0.752 | 0.824 |
| | Acierto por carácter | 0.888 | 0.919 | 0.898 | 0.931 |
| Sobre recorte anotado (techo) | n | 208 | 208 | 208 | 208 |
| | Exact match | 0.716 | 0.788 | 0.736 | 0.822 |
| | Acierto por carácter | 0.881 | 0.924 | 0.885 | 0.932 |

Diferencias del VLM frente a PaddleOCR, en puntos porcentuales:

| Escenario | Exact match (crudo) | Exact match | Por carácter | Por carácter |
|---|---|---|---|---|
| Detecciones | +7,2 | +7,2 | +3,1 | +3,3 |
| Techo | +7,2 | +8,6 | +4,3 | +4,7 |

Mejora que aporta la corrección por posición sobre cada lector:

| Escenario | PaddleOCR | VLM |
|---|---|---|
| Detecciones | +3,8 | +3,8 |
| Techo | +2,0 | +3,4 |

Los resultados muestran que VLM obtiene un rendimiento superior a PaddleOCR en ambos escenarios de evaluación, tanto en exact match como en acierto por carácter. Sobre las detecciones de YOLO, VLM alcanza un exact match de 0,786 frente a 0,714 de PaddleOCR, mientras que el acierto por carácter aumenta de 0,888 a 0,919. Esta ventaja se mantiene al aplicar la corrección, con valores de 0,824 y 0,931 para VLM, frente a 0,752 y 0,898 de PaddleOCR. En el escenario de recorte anotado, se observa la misma tendencia: VLM obtiene un exact match de 0,788 y un acierto por carácter de 0,924, superando los 0,716 y 0,881 de PaddleOCR, respectivamente.

Tras la corrección, ambos modelos mejoran sus resultados, alcanzando VLM 0,822 de exact match y 0,932 de acierto por carácter, frente a 0,736 y 0,885 de PaddleOCR. En términos generales, la corrección aporta mejoras consistentes en ambos modelos y escenarios, aunque no modifica la ventaja relativa de VLM. Sobre las detecciones aporta lo mismo a ambos lectores, mientras que sobre el techo beneficia más al VLM. Esta ventaja resaltada, podría estar relacionada con la arquitectura del VLM basada en la combinación de un codificador visual con un modelo de lenguaje.

### 2.3 Tiempo por lectura

| Medición | PaddleOCR (media / mediana) | VLM (media / mediana) | Factor (media) | Factor (mediana) |
|---|---|---|---|---|
| Detecciones | 415 ms / 333 ms | 540 ms / 514 ms | ~1,3x | ~1,5x |
| Recorte anotado (techo) | 425 ms / 338 ms | 538 ms / 512 ms | ~1,3x | ~1,5x |

En términos de rendimiento temporal, PaddleOCR presenta una ventaja frente a VLM en ambos escenarios de evaluación, aunque con YOLO esta diferencia es moderada. Sobre las detecciones de YOLO, PaddleOCR obtiene un tiempo medio de procesamiento de 415 ms, frente a los 540 ms de VLM, lo que supone que VLM necesita aproximadamente 1,3 veces más tiempo para realizar el reconocimiento. En el escenario de recorte anotado, los valores son prácticamente equivalentes: PaddleOCR registra 425 ms, mientras que VLM alcanza los 538 ms, con un factor de aproximadamente 1,3 en la media. En ambos escenarios, las medianas mantienen una diferencia similar, con un factor aproximado de 1,5. Cabe señalar que el VLM se ejecuta en GPU y PaddleOCR en CPU, por lo que la comparación de tiempos no se realiza sobre el mismo hardware.

## 3. Cruce detector × OCR (todos los cuadernos)

| Combinación | Detecciones | Placas halladas | Falsos positivos (IoU < 0.5) | Exact match (detecciones) | Con corrección | Por carácter | Tiempo medio |
|---|---|---|---|---|---|---|---|
| YOLO + PaddleOCR | 376 | 210 | 55 | 0.714 | 0.752 | 0.888 | 415 ms |
| YOLO + VLM | 376 | 210 | 55 | 0.786 | 0.824 | 0.919 | 540 ms |
| RT-DETR + PaddleOCR (conf 0.35) | 399 | 199 | 136 | 0.714 | 0.754 | 0.889 | 416 ms |
| RT-DETR + VLM (conf 0.35) | 399 | 199 | 136 | 0.814 | 0.854 | 0.932 | 777 ms |

- El techo es idéntico con YOLO y con RT-DETR para cada lector, como cabe esperar al no intervenir el detector.
- Con RT-DETR, PaddleOCR y VLM evalúan exactamente las mismas 399 detecciones, así que esa comparación es directa.
- El exact match sobre detecciones no permite declarar un detector ganador: los conjuntos evaluados difieren. RT-DETR + VLM sale algo por encima de YOLO + VLM, pero sobre menos placas, y con PaddleOCR el resultado es idéntico.
- La diferencia de detección sí es clara: YOLO encuentra más placas con muchos menos falsos positivos y menos detecciones totales.

El cruce confirma que la mejora del VLM sobre PaddleOCR se mantiene con ambos detectores: con YOLO el exact match pasa de 0.714 a 0.786 y con RT-DETR de 0.714 a 0.814. En cambio, el tiempo medio de PaddleOCR apenas varía entre detectores (415 ms frente a 416 ms), mientras que el del VLM es notablemente menor con YOLO que con RT-DETR. En cuanto a la elección del detector, la diferencia más clara no está en la lectura sino en la detección, donde YOLO encuentra más placas con menos falsos positivos.

## 4. Conclusiones

Estos resultados indican que VLM lee mejor que PaddleOCR con el mismo detector: la ventaja es de +7,2 puntos de exact match tanto sobre las detecciones como sobre el techo, y de +3,1 y +4,3 puntos por carácter respectivamente. Con corrección por posición, la ventaja se mantiene o crece. La corrección por posición aporta de 2,0 a 3,8 puntos de exact match, y beneficia por igual a ambos lectores sobre las detecciones, mientras que sobre el techo favorece más al VLM.

En cuanto al coste computacional, con YOLO este es moderado: el VLM es ~1,3x más lento de media y ~1,5x de mediana, teniendo en cuenta que el VLM se ejecuta en GPU y PaddleOCR en CPU. Por tanto, con YOLO como detector, la mayor precisión de reconocimiento del VLM se obtiene con un incremento de tiempo mucho más contenido que con RT-DETR, lo que lo convierte en una alternativa viable cuando la exactitud es prioritaria. PaddleOCR sigue siendo la opción más rápida cuando la velocidad de respuesta y la eficiencia temporal son los factores determinantes.

Por último, en la comparación entre detectores, YOLO es el más limpio: presenta 55 falsos positivos geométricos sobre 376 detecciones, además de encontrar más placas que RT-DETR.
