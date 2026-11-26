En este repositorio se encontrarán distintas tareas prácticas de **Visión por Computador**, con el objetivo de aprender acerca
de Anaconda, Python y la librería OpenCV.

### 👥 Participantes de este repositorio:
- [Enrique Sosa Ojeda](https://github.com/Enric1005)
- [Vidal de León Giménez](https://github.com/t3ntox)

------

### [PRÁCTICA 1](https://github.com/t3ntox/Practicas_VC/tree/main/P1)

En esta primera práctica se encontrará 4 tareas distintas, todas ellas desarrolladas dentro de un mismo cuaderno Jupyter.
Además, del cuaderno Jupyter, encontrarán en la carpeta destinada a esta práctica dos imágenes, que corresponden a dos de los mondrian generados
con código en una de las tareas.

Para la realización de dichas tareas, hemos hecho uso de herramientas IA cuyas conversaciones se encuentran adjuntas en el cuaderno y de la guía aportada para la P1 en el repositorio de la [asignatura](https://github.com/otsedom/otsedom.github.io/tree/main/VC).

Estructura de la carpeta
```
P1
|- imagen.jpg            #Ejemplo de mondrian 1
|- imagen1.jpg           #Ejemplo de mondrian 2
|- tareas_p1.ipynb       #Cuaderno jupyter con tareas desarrolladas
```


### [PRÁCTICA 2](https://github.com/t3ntox/Practicas_VC/tree/main/P2)

En esta segunda práctica se encontrará 3 tareas distintas, todas ellas desarrolladas dentro de un mismo cuaderno Jupyter.
Además, del cuaderno Jupyter, encontrarán en la carpeta destinada a esta práctica una imágen, que corresponde a la imagen utilizada para realizar distintos tratamientos de imagen distintos, y un archivo que corresponde al modelo usado en la última tarea para la detección de rostros.

Para la realización de dichas tareas, se incluye en la carpeta de esta práctica un [README](https://github.com/t3ntox/Practicas_VC/tree/main/P2/README.md) donde se desarrolla el proceso de realización de cada tarea.

Estructura de la carpeta
```
P2
|- face_detection_yunet_2023mar.onnx        #Modelo de dectección de rostro
|- mandril.jpg                              #Imagen base para la realización de las tareas
|- README.md
|- Resultado_ampliacion_Tarea_2.gif         
|- Resultado_Ampliación.gif                 #Demostración del resultado de detección de rostros con YuNet
|- Resultado_escenario_1.png
|- Resultado_escenario_2.png
|- Resultado_Tarea_1.png
|- Resultado_Tarea_2.png
|- tareas_p2.ipynb                          #Cuaderno jupyter con tareas desarrolladas
```


### [PRÁCTICA 3](https://github.com/t3ntox/Practicas_VC/tree/main/P3)

En esta tercera práctica se encontrará 2 tareas distintas, todas ellas desarrolladas dentro de un mismo cuaderno Jupyter.
Además, del cuaderno Jupyter, encontrarán en la carpeta destinada a esta práctica otra carpeta destinada a mostrar los resultados de los distintos pasos llevados a cabo en la detección de plásticos.

Para la realización de dichas tareas, se incluye en la carpeta de esta práctica un [README](https://github.com/t3ntox/Practicas_VC/tree/main/P3/README.md) donde se desarrolla el proceso de realización de cada tarea junto con imágenes y gif como muestra del resultado del código desarrollado.

Estructura de la carpeta
```
P3
|- resultados/                                #Resultados obtenidos a lo largo del proceso de detección de los plásticos
|- FRA.png       
|- MPs_test.jpg                             
|- Monedas.jpg                                #Imágenes de monedas para los primeros pasos en la detección de monedas
|- Monedas1.jpg       
|- Monedas2.jpg               
|- PEL.png
|- README.md
|- Resultado_tarea_1.gif                      #Demostración del funcionamiento del detector de monedas
|- Resultado_tarea_previa.png                 #Resultado de detección en la tarea previa del detector de monedas
|- TAR.png
|- fragment-03-olympus-10-01-2020.JPG
|- pellet-03-olympus-10-01-2020.JPG
|- tar-03-olympus-10-01-2020.JPG
|- tareas_p3.ipynb
```

### [PRÁCTICA 4](https://github.com/t3ntox/Practicas_VC/tree/main/P4)

En esta cuarta práctica se abordará el desarrollo y comparación de YOLO y RT-DETR, así como la integración de herramientas de OCR como PaddleOCR y modelos de lenguaje de VLM.

Para la realización de dichas tareas, se incluye el código en diferentes cuadernos Jupyter junto con scripts de utilidad y carpetas dedicadas al almacenamiento de pesos, resultados de entrenamiento y evaluaciones.

Estructura de la carpeta
```
P4
|- comparativas/                                   # Comparativas entre los diferentes modelos y resultados
|- ocr_eval/                                       
|- plates/
|- resultados/                                     # Comparativa de detecciones en imágenes entre YOLO Y RT-DETR                                         
|- RT-DETR/                                       
|- RT-DETR_PaddleOCR_result/                       
|- RT-DETR_resultado_entrenamiento/                
|- RT-DETR_VLM_result/                             
|- vlm_eval/                                       
|- weights/                                        
|- YOLO/                                           
|- YOLO_PaddleOCR_result/                          
|- YOLO_resultado_entrenamiento/                   
|- YOLO_VLM_result/                                
|- C0142.MP4                                       # Vídeo de prueba empleado para la detección en movimiento
|- load_rtdetr.py                                  
|- RT-DETR_entrenamiento.ipynb                     
|- RT-DETR_PaddleOCR.ipynb                         # Cuaderno Jupyter para la integración de RT-DETR y PaddleOCR
|- RT-DETR_VLM.ipynb                               # Cuaderno Jupyter para la integración de RT-DETR con VLM
|- utils_matriculas.py                             
|- YOLO_entrenamiento.ipynb                        
|- YOLO_PaddleOCR.ipynb                            # Cuaderno Jupyter para la integración de YOLO y PaddleOCR
|- YOLO_VLM.ipynb                                  # Cuaderno Jupyter para la integración de YOLO con VLM
|- yolo11l.pt                                      
```

### [PRÁCTICA 5](https://github.com/t3ntox/Practicas_VC/tree/main/P5)

En esta quinta práctica se construye un modelo biométrico propio capaz de reconocer 7 expresiones faciales (`angry`, `duckface`, `fear`, `happy`, `neutral`, `sad` y `surprise`). Para ello se entrenan cinco clasificadores clásicos (HOG, LBP y autocaras con SVM o KNN) y una red convolucional propia tipo mini-Xception, que se comparan entre sí y con DeepFace. El modelo elegido, la CNN, se aplica después en dos demostradores en tiempo real con webcam: reacciones de partidos políticos según la emoción y un filtro satírico de "derechos" por país.

El código se reparte en varios cuadernos Jupyter (preparación del dataset, un cuaderno por modelo, la comparativa y los dos demostradores) que comparten las funciones del script `comun.py`. Se incluye en la carpeta de esta práctica un [README](https://github.com/t3ntox/Practicas_VC/tree/main/P5/README.md) donde se desarrolla el dataset, el entrenamiento y los resultados de cada modelo, la comparativa y el funcionamiento de los demostradores, junto con los diagramas de flujo de la CNN.

Estructura de la carpeta
```
P5
|- banderas/                         #Banderas de los países (caché del selector)
|- dataset/                          #Dataset de emociones (train/test) y Selfie-dataset
|- emotion-images/                   #Imágenes de reacción por partido y emoción
|- informe_entrenamiento/            #Gráficas, tablas y resúmenes del entrenamiento y la comparativa
|- logos-partidos/                   #Logotipos del selector de partidos
|- modelos/                          #Modelos entrenados y modelos preentrenados
|- comparativa.ipynb                 #Comparativa de todos los modelos y frente a DeepFace
|- conclusiones_comparativas.md      #Explicación de los datos de la comparativa
|- comun.py                          #Funciones comunes a todos los cuadernos
|- emociones_políticas.ipynb         #Demostrador de reacciones políticas
|- entrenar_01_hog_svm.ipynb         #Entrenamiento HOG + SVM
|- entrenar_02_lbp_svm.ipynb         #Entrenamiento LBP + SVM
|- entrenar_03_hoglbp_pca_svm.ipynb  #Entrenamiento HOG + LBP + PCA + SVM
|- entrenar_04_autocaras_pca_svm.ipynb #Entrenamiento autocaras (PCA) + SVM
|- entrenar_05_hog_pca_knn.ipynb     #Entrenamiento HOG + PCA + KNN
|- entrenar_06_cnn.ipynb             #Entrenamiento de la CNN propia
|- filtro_derechos.ipynb             #Demostrador del filtro de derechos
|- paises.csv                        #Factores por país del filtro de derechos
|- preparar_dataset.ipynb            #Preparación de la clase duckface
|- README.md
|- selector.py                       #Selector lateral de imágenes
```
