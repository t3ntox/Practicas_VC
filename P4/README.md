# Práctica 4

En este directorio del repositorio encontrarán múltiples cuadernos jupyter cada uno destinado a una serie de tareas concretas. Los distintos archivos son:
- `RT-DETR_entrenamiento.ipynb`, `YOLO_entrenamiento.ipynb` -> destinados a la ejecución del entrenamiento junto a las gráficas que representan los distintos datos extraidos para una posterior comparativa.
- `RT-DETR_PaddleOCR.ipynb`, `YOLO_PaddleOCR.ipynb` -> destinados a la evaluación del ocr clásico seleccionado, junto a su aplicación en el vídeo.
- `RT-DETR_VLM.ipynb`, `YOLO_VLM.ipynb` -> destinados a la evaluacion del VLM seleccionado, Hugging Face, junto a su aplicación en el vídeo.

Todos los cuadernos, menos los dedicados al entrenamiento, dependen del archivo `utils_matrículas.py` que recoge todas las funciones comunes a los cuadernos para evitar la duplicidad en el código. 

### 👥 Autores:
- [Enrique Sosa Ojeda](https://github.com/Enric1005)
- [Vidal de León Giménez](https://github.com/t3ntox)

------


## Entrenamiento
Para el entrenamiento contamos con un dataset de 2812 imágenes. Dicho dataset se encuentra: https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI?usp=sharing 

### [YOLO](./YOLO_entrenamiento.ipynb)


### [RT-DETR](./RT-DETR_entrenamiento.ipynb)
Para el entrenamiento de este modelo de código abierto no partimos del dataset en crudo, sino que se procesa a través del archivo [dataset_manager.py](./plates/dataset_manager.py). Dicho archivo, es el encargado de generar 2 archivos .json que recogen las anotaciones por un lado de entrenamiento(75%), y por otro lado de validación(25%). Estas anotaciones recogidas, pasan de tener un formato relativo normalizado a un formato de píxeles absolutos, formato COCO. Además, hace un procesamiento de imágenes limitando el tamaño del lado mayor manteniendo el ratio de aspecto, corrigiendo la orientación física de las fotos (`EXIF`) y unificando la extensión de las imágenes.
Los resultados de estas operaciones aparecerán en un directorio llamado `coco-dataset`.

Para realizar el entrenamiento al modelo inicial se le han añadido una serie de archivos de configuracion, para poder personalizar epocas, lr, warmup,... Las condiciones principales en las que se ha realizado el entrenamiento son:
- 50 épocas
- warmup de 100 iteraciones
- lr 0.0001 y 0.00001 para el backbone
- Durante las últimas 3 épocas no se realizan ningún tipo de transformacion a las imágenes

El modelo de entrenamiento seleccionado, es el `s` de la versión 2 de RT-DETR (se puede ver en [load_rtdetr.py](./load_rtdetr.py) específicamente el modelo que corresponde a `s`). Este parte de sus pesos originales en el entrenamiento, no se realiza la congelación de capas del backbone para optimizar el entrenamiento, que a través de los datos con los que se entrena se van adaptando a la única clase que se quiere detectar `matrículas`.

------

## Pipeline utilizado
Para ambos modelos se emplean los mismos pasos para el tratamiento del vídeo:
1. Se realiza la detección de las clases principales con los modelos principales, `yolol` y `r-tdetrl`.
2. A partir de la detección de vehículos(coches, camiones, motos y guaguas) se le pasa un recorte de la detección al modelo entrenado para las matrículas.
3. Estos modelos entrenados realizan la detección de la matrícula en el recorte, realizando un nuevo recorte para que el OCR clásico o el VLM seleccionado puedan, tras aplicar un filtro gaussiano al recorte, tratar de leer la matrícula.
4. Tras la lectura de la matrícula discriminamos resultados haciendo uno de regex, los cuales representan los patrones que pueden seguir las distintas matrículas de los distintos países europeos.

Para todo este proceso se aplican; un sistema de votación para evitar la clasificación erronea de vehículos (como puede pasar con un SUV), un histórico de las detecciones para poder unificar sus id en caso de cambio a lo largo del track, y en el caso de las matrículas se exige un tamaño mínimo para su lectura junto a una discriminación en base al producto de la confianza del detector de matrículas y la confianza del modelo de lectura. 

Únicamente para RT-DETR, con el objetivo de obtener un track lo más preciso posible, se emplean dos trackers distintos: uno para personas y otros para vehículos. Esta distinción, permite que el parámetro `lost_track_buffer` sea distinto ya que ambas clases se mueven en pantalla a velocidades muy distintas. Por ello, el tracker de las personas requiere un valor más alto, ya que sus oclusiones se prolongan más en el tiempo, para mantener el id durante el track. 

Vídeos:
- [Video YOLO PaddleOCR](./YOLO_PaddleOCR_result/resultado.mp4)
- [Video RT-DETR PaddleOCR](./RT-DETR_PaddleOCR_result/resultado.mp4)
- [Video YOLO Hugging Face](./YOLO_VLM_result/resultado.mp4)
- [Video RT-DETR Hugging Face](./RT-DETR_VLM_result/resultado.mp4)

------

## [Comparativas](./comparativas/)
Se encontrarán también con comparativas entre OCR clásico y VLM, estás comparaciones se han hecho con base en [RT-DETR](./comparativas/comparativa_rtdetr_ocr_vs_vlm.md) y [YOLO](./comparativas/comparativa_yolo_ocr_vs_vlm.md). En ellas, se extrae la precisión en la lectura de matrículas a partir de las detecciones de matrículas de sus repectivos modelos usados como base y a partir de las anotaciones del propio dataset. Junto a ello, también se calcula a partir de la distancia de Levenshtein la precision caracter por caracter del modelo. Por último, también se extrae el tiempo que tardan por detección de media los modelos.
Otra de las comparativas realizadas, es entre la [detección de matrículas](./comparativas/comparativa_yolo_vs_rtdetr.md) con YOLO y con RT-DETR. En dicha comparativa, a través de las gráficas extraídas en el entrenamiento se podrá observar la tolerancia al cambio de la confianza, la progresión realizada durante los entrenamientos, entre otras cuestiones que se tratan en el archivo dedicado.

------

Durante la realización del código de esta práctica se ha hecho uso de la IA Claude.