# Práctica 2

Este directorio contiene el cuaderno `tareas_p2.ipynb`, que aborda técnicas de detección de bordes (Canny y Sobel) sobre la imagen `mandril.jpg`, y una aplicación práctica de detección de rostros en tiempo real con el modelo YuNet.

## Tareas

### Tarea 1 — Conteo de píxeles blancos por filas (Canny)

Aplica el detector de bordes de **Canny** sobre la imagen en escala de grises y realiza el conteo de píxeles blancos (bordes detectados) **por filas**, en lugar de por columnas como en el ejemplo de partida.

- Se calcula `maxfil`, el valor máximo de píxeles blancos en una fila.
- Se identifican y muestran las filas cuyo conteo supera el 90% de `maxfil`.
- Las filas destacadas se resaltan con líneas horizontales rojas sobre la imagen de Canny.
- Se incluye una gráfica adicional con el porcentaje de píxeles blancos por fila, para visualizar la distribución de bordes a lo largo de la imagen.

**Conclusión:** las filas superiores de la imagen (zona del pelaje con mayor contraste) concentran más bordes detectados, mientras que las filas inferiores, con un pelaje más uniforme, requerirían umbrales de Canny más bajos para detectar bordes de forma equivalente.

### Tarea 2 — Umbralizado de Sobel y comparación con Canny

Se aplica el operador **Sobel** (combinando gradiente horizontal y vertical, tras un suavizado gaussiano previo) sobre la imagen, y se umbraliza el resultado a binario.

- Se realiza el conteo de píxeles no nulos por filas y por columnas sobre la imagen umbralizada de Sobel, siguiendo la misma lógica de la Tarea 1.
- Se calculan los máximos de fila y columna, y se destacan aquellas posiciones que superan el 90% de sus respectivos máximos.
- Se muestran, lado a lado, los resultados de **Canny** (resaltado en verde) y **Sobel umbralizado** (resaltado en rojo) sobre la imagen del mandril, incluyendo el conteo total de filas y columnas destacadas de cada método.

**Conclusión:** ambos métodos coinciden en gran medida en qué filas destacan, pero Sobel (tras umbralizar) detecta un número significativamente mayor de filas relevantes. En cambio, para las columnas, es Canny quien detecta un número mayor de columnas destacadas, aunque también existe coincidencia entre ambos métodos.

### Tarea 3 — Demostrador: detección de rostros con anonimización en tiempo real

Inspirado en la instalación *[My Little Piece of Privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy)* de Niklas Roy, se ha implementado un demostrador que protege la privacidad de las personas frente a una cámara:

- Utiliza el modelo preentrenado **YuNet** (`cv2.FaceDetectorYN`) para detectar rostros en tiempo real a través de la webcam.
- Por cada rostro detectado, su región (bounding box) se sustituye por una imagen definida de antemano (`mandril.jpg`), redimensionada al tamaño del rostro, ocultando así la identidad de la persona.
- El bounding box se ajusta (clip) a los límites del frame para evitar errores por detecciones cercanas a los bordes de la imagen.
- La aplicación funciona en bucle sobre el flujo de vídeo hasta que el usuario pulsa la tecla `Esc`.

**Idea de la instalación original:** al igual que la cortina física del proyecto de Niklas Roy se movía para bloquear la visión hacia el interior del local al detectar presencia humana, este demostrador "tapa" digitalmente cada rostro detectado, preservando la privacidad visual en tiempo real.


## Archivos necesarios

- `mandril.jpg` — imagen base utilizada en las Tareas 1, 2 y 3 (como imagen de "tapado" de rostros).
- `face_detection_yunet_2023mar.onnx` — modelo preentrenado de detección de rostros, necesario para la Tarea 3 (descargable desde [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).
