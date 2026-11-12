# Comparativa de gráficas del detector de matrículas: YOLO vs RT-DETR

## 1. Resumen de métricas

| Métrica | YOLO | RT-DETR | Diferencia |
|---|---|---|---|
| mAP@0.5 | **0.776** | 0.667 | +0,109 a favor de YOLO |
| F1 máximo | **0.82** (conf 0.409) | 0.74 (conf 0.491) | +0,08 |
| Recall máximo (conf ≈ 0) | **0.83** | 0.76 | +0,07 |
| Precisión 1.00 alcanzada en conf | 0.841 | 0.950 | RT-DETR necesita ~0,11 más |
| Recall en matriz normalizada | **0.87** | 0.77 | +0,10 |
| Falsos negativos (normalizado) | 0.13 | 0.23 | RT-DETR pierde 0,10 más (casi el doble) |
| Verdaderos positivos (matriz absoluta) | 846 | 740 | YOLO +106 (RT-DETR detecta un 12,5 % menos) |
| Falsos negativos (matriz absoluta) | 132 | 227 | RT-DETR +95 (un 72 % más de placas perdidas) |
| Falsos positivos (matriz absoluta) | 9 772 | 119 260 | RT-DETR ~12,2 veces más (+109 488) |
| Placas anotadas (TP + FN) | 978 | 967 | 11 de diferencia entre conjuntos |

## 2. Análisis por gráfica

### 2.1 Curva F1-Confianza

| YOLO | RT-DETR |
|---|---|
| ![F1 YOLO](./YOLO_resultados_entrenamiento/BoxF1_curve.png) | ![F1 RT-DETR](./RT-DETR_resultado_entrenamiento/F1_curve.png) |

- **YOLO**: sube rápido y forma una meseta muy ancha, con F1 ≈ 0.80 o más entre conf ~0.3 y ~0.6 (**~**). El máximo (0.82) está en 0.409 y cae con fuerza a partir de ~0.65.
- **RT-DETR**: forma de cúpula más estrecha, con máximo 0.74 en 0.491. Por encima de 0.7 solo entre conf ~0.35 y ~0.68 (**~**), y llega a cero cerca de 0.93.
- **Lectura**: YOLO es más tolerante a la elección del umbral; RT-DETR es más sensible a él y su techo es más bajo.

### 2.2 Curva Precisión-Confianza

| YOLO | RT-DETR |
|---|---|
| ![Precisión YOLO](./YOLO_resultados_entrenamiento/BoxP_curve.png) | ![Precisión RT-DETR](./RT-DETR_resultado_entrenamiento/P_curve.png) |

- **YOLO**: precisión ~0.8 con conf 0.2, ~0.9 con 0.4 y ~0.94 con 0.65 (**~**). Llega a 1.00 en 0.841.
- **RT-DETR**: sube más despacio, con ~0.5 en conf 0.2, ~0.8 en 0.45 y ~0.9 en 0.7 (**~**). Llega a 1.00 solo en 0.950, con picos ruidosos en la clase `plate` cerca de 0.9.
- **Lectura**: las confianzas de RT-DETR están peor calibradas a umbrales bajos: para igualar la precisión de YOLO hay que subir el umbral bastante.

### 2.3 Curva Recall-Confianza

| YOLO | RT-DETR |
|---|---|
| ![Recall YOLO](./YOLO_resultados_entrenamiento/BoxR_curve.png) | ![Recall RT-DETR](./RT-DETR_resultado_entrenamiento/R_curve.png) |

- **YOLO**: parte de 0.83, baja suavemente a ~0.77 en conf 0.2 y ~0.70 en 0.6, y colapsa entre 0.7 y 0.85 (**~**).
- **RT-DETR**: parte de 0.76, baja a ~0.73 en 0.2 y ~0.63 en 0.6, y colapsa entre 0.8 y 0.92 (**~**).
- **Lectura**: RT-DETR genera confianzas más altas (su curva se desplaza a la derecha), pero con menos recall en todo el rango.

### 2.4 Curva Precisión-Recall

| YOLO | RT-DETR |
|---|---|
| ![PR YOLO](./YOLO_resultados_entrenamiento/BoxPR_curve.png) | ![PR RT-DETR](./RT-DETR_resultado_entrenamiento/PR_curve.png) |

- **YOLO** (mAP50 0.776): precisión ~0.99 → ~0.94 hasta recall ~0.7 y caída brusca entre 0.77 y 0.85, donde termina (**~**).
- **RT-DETR** (mAP50 0.667): precisión ~0.95 → ~0.90 hasta recall ~0.6 y caída brusca en torno a 0.70-0.76, donde termina (**~**).
- **Lectura**: la curva de YOLO envuelve a la de RT-DETR. Es más alta (mejor precisión) y llega más lejos (mayor recall máximo), de ahí los 0,109 de diferencia en mAP50.

### 2.5 Matrices de confusión

| | YOLO | RT-DETR |
|---|---|---|
| Absoluta | ![Matriz YOLO](./YOLO_resultados_entrenamiento/confusion_matrix.png) | ![Matriz RT-DETR](./RT-DETR_resultado_entrenamiento/confusion_matrix.png) |
| Normalizada | ![Matriz normalizada YOLO](./YOLO_resultados_entrenamiento/confusion_matrix_normalized.png) | ![Matriz normalizada RT-DETR](./RT-DETR_resultado_entrenamiento/confusion_matrix_normalized.png) |

| | YOLO | RT-DETR |
|---|---|---|
| Recall (placa detectada) | 0.87 | 0.77 |
| Placas perdidas (FN) | 132 (13 %) | 227 (23 %) |
| Falsos positivos | 9 772 | 119 260 |
| Ratio FP / TP | ~11,6 | ~161 |

Los falsos positivos son muy altos en ambos modelos. Esto indica que las matrices se calcularon con un umbral de confianza muy bajo (típico de validación) y **no representan el punto de trabajo real**. RT-DETR produce una cantidad fija de predicciones por imagen (consultas del decodificador), lo que explica en parte que se dispare tanto. Sirven para comparar recall, no precisión de uso.

## 3. Relación con los cuadernos de OCR

| Detector | Umbral óptimo F1 (gráficas) | Umbrales usados en los cuadernos |
|---|---|---|
| YOLO | 0.409 | Evaluación 0.25; vídeo `CONF_PLACA` 0.4 |
| RT-DETR | 0.491 | Evaluación 0.35 (PaddleOCR) y 0.25 (VLM); vídeo `PLATE_CONF` 0.3 |

- En vídeo, YOLO ya trabaja casi en su óptimo (0.4 frente a 0.409).
- RT-DETR trabaja por debajo de su óptimo en todos los casos. Esto es coherente con lo visto en las comparativas anteriores, donde RT-DETR producía muchos más falsos positivos geométricos (278 con conf 0.35, frente a 108 de YOLO con conf 0.25), localizando el mismo número de placas (188/407).
- A igual umbral (0.25) en la evaluación con el VLM, RT-DETR generó 613 detecciones frente a 367 de YOLO y encontró menos placas (182 frente a 188 de 407), en línea con su peor recall y su menor precisión a umbrales bajos.

## 4. Conclusiones

1. **YOLO supera a RT-DETR en todas las métricas de detección**: mAP50 (+0,109), F1 (+0,08), recall (+0,07 a +0,10) y menos placas perdidas (13 % frente a 23 %).
2. **YOLO es más robusto al umbral**: meseta de F1 más ancha y mejor calibración de la confianza.
3. **RT-DETR requiere umbrales más altos** (~0.5 para el óptimo de F1) y sus detecciones de alta confianza son fiables solo por encima de ~0.9.
4. **Recomendación**: subir el umbral de RT-DETR hacia 0.45-0.5 en los cuadernos para reducir falsos positivos antes del OCR, y volver a medir. Es una sugerencia basada en las curvas, no un resultado verificado en el pipeline.

## 5. Advertencias

- Las dos matrices indican conjuntos de validación ligeramente distintos (978 vs 967 placas anotadas), o criterios de emparejamiento diferentes, por lo que la comparación es muy cercana pero no idéntica.
- Solo se dispone de mAP@0.5, no de mAP@0.5:0.95, así que no se puede comparar la precisión de las cajas.
- Asigné cada PDF a su modelo según tu indicación (`graficas.pdf` = YOLO) y el estilo de las gráficas.
