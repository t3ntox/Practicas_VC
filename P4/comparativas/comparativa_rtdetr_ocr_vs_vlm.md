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

Los resultados muestran que VLM obtiene un rendimiento superior a PaddleOCR en ambos escenarios de evaluación, tanto en exact match como en acierto por carácter. Sobre las detecciones de RT-DETR, VLM alcanza un exact match de 0,814 frente a 0,714 de PaddleOCR, mientras que el acierto por carácter aumenta de 0,889 a 0,932. Esta ventaja se mantiene al aplicar la corrección, con valores de 0,854 y 0,942 para VLM, frente a 0,754 y 0,9 de PaddleOCR. En el escenario de recorte anotado, se observa la misma tendencia: VLM obtiene un exact match de 0,788 y un acierto por carácter de 0,924, superando los 0,716 y 0,881 de PaddleOCR, respectivamente. 
Tras la corrección, ambos modelos mejoran sus resultados, alcanzando VLM 0,822 de exact match y 0,932 de acierto por carácter, frente a 0,736 y 0,885 de PaddleOCR. En términos generales, la corrección aporta mejoras consistentes en ambos modelos y escenarios, aunque no modifica la ventaja relativa de VLM. Esta ventaja resaltada, podría estar relacionada con la arquitectura del VLM basada en la combinación un codificador visual con un modelo de lenguaje

### 2.3 Tiempo por lectura

| Escenario | PaddleOCR (media / mediana) | VLM (media / mediana) | Factor (media) | Factor (mediana) |
|---|---|---|---|---|
| Detecciones | 416 ms / 330 ms | 777 ms / 519 ms | ~1,9x | ~1,6x |
| Recorte anotado (techo) | 417 ms / 333 ms | 871 ms / 517 ms | ~2,1x | ~1,6x |

En términos de rendimiento temporal, PaddleOCR presenta una ventaja clara frente a VLM en ambos escenarios de evaluación. Sobre las detecciones de RT-DETR, PaddleOCR obtiene un tiempo medio de procesamiento de 416 ms, frente a los 777 ms de VLM, lo que supone que VLM necesita aproximadamente 1,9 veces más tiempo para realizar el reconocimiento.  En el escenario de recorte anotado, la diferencia es todavía mayor en términos de media: PaddleOCR registra 417 ms, mientras que VLM alcanza los 871 ms, situándose su tiempo de ejecución en aproximadamente 2,1 veces el de PaddleOCR. Sin embargo, las medianas mantienen una diferencia similar a la del escenario anterior, con un factor aproximado de 1,6. 

## 3. Conclusiones

Estos resultados indican que, aunque VLM ofrece una mayor precisión de reconocimiento, esta mejora implica un coste computacional y temporal considerablemente superior. Por el contrario, PaddleOCR presenta una ejecución más rápida y tiempos de procesamiento más reducidos en ambos escenarios, lo que lo convierte en una alternativa más adecuada cuando la velocidad de respuesta y la eficiencia temporal son factores prioritarios.
