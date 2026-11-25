import os
import threading
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

DIR_MODELOS = Path(__file__).resolve().parent / "modelos"

DNN_PROTO = str(DIR_MODELOS / "deploy.prototxt.txt")
DNN_PESOS = str(DIR_MODELOS / "res10_300x300_ssd_iter_140000.caffemodel")
LANDMARKER = str(DIR_MODELOS / "face_landmarker.task")
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")

RECORTE_CARA = 0.05

EXTENSIONES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class DetectorCaras:
    def __init__(self, conf_min=0.5, proto=DNN_PROTO, pesos=DNN_PESOS):
        if not hasattr(cv2.dnn, "readNetFromCaffe"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} no puede leer modelos Caffe (se quitó en OpenCV 5). "
                "Instala OpenCV 4.x con un solo paquete: pip uninstall -y opencv-python "
                "opencv-contrib-python opencv-contrib-python-headless && pip install opencv-contrib-python==4.12.0.88")
        self.net = cv2.dnn.readNetFromCaffe(np.fromfile(proto, np.uint8), np.fromfile(pesos, np.uint8))
        self.conf_min = conf_min

    def detectar(self, img, min_lado=20):
        H, W = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104.0, 117.0, 123.0], False, False)
        self.net.setInput(blob)
        det = self.net.forward()
        cajas = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < self.conf_min:
                continue
            x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([W, H, W, H])).astype(int)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if x2 - x1 >= min_lado and y2 - y1 >= min_lado:
                cajas.append((x1, y1, x2 - x1, y2 - y1, conf))
        return cajas

    def mayor(self, img):
        cajas = self.detectar(img)
        if not cajas:
            return None
        return max(cajas, key=lambda c: c[2] * c[3])[:4]


def recortar_cara(img, caja, recorte=RECORTE_CARA):
    x, y, w, h = caja
    mx, my = int(w * recorte), int(h * recorte)
    H, W = img.shape[:2]
    x0, y0 = max(x + mx, 0), max(y + my, 0)
    x1, y1 = min(x + w - mx, W), min(y + h - my, H)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return img[y0:y1, x0:x1]

class Landmarks:
    OJO_IZQ = (33, 133)       # comisuras del ojo izquierdo de la imagen
    OJO_DER = (263, 362)      # comisuras del ojo derecho de la imagen
    FRENTE = 10               # parte alta de la frente
    MENTON = 152
    NARIZ = 1
    BOCA = (61, 291, 13, 14)  # comisuras y labios superior/inferior

    def __init__(self, ruta=LANDMARKER, descargar=True, video=True):
        # video=True: modo vídeo (sigue la cara entre frames, para la webcam);
        # video=False: cada imagen se procesa por separado (imágenes sueltas).
        self.ok, self._t, self.video = False, 0, video
        try:
            if not os.path.exists(ruta) and descargar:
                print("Descargando", LANDMARKER_URL)
                urllib.request.urlretrieve(LANDMARKER_URL, ruta)
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._mp = mp
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=open(ruta, "rb").read()),
                running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
                num_faces=1)
            self.modelo = vision.FaceLandmarker.create_from_options(opts)
            self.ok = True
        except Exception as e:
            print("Landmarks desactivados:", e)

    def detectar(self, frame_bgr):
        if not self.ok:
            return None
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            self._t = max(self._t + 1, int(time.time() * 1000))    
            res = self.modelo.detect_for_video(img, self._t)
        else:
            res = self.modelo.detect(img)
        if not res.face_landmarks:
            return None
        return np.array([(p.x * W, p.y * H) for p in res.face_landmarks[0]], np.float32)

    @classmethod
    def ojos(cls, pts):
        return pts[list(cls.OJO_IZQ)].mean(0), pts[list(cls.OJO_DER)].mean(0)

    @classmethod
    def angulo(cls, pts):
        a, b = cls.ojos(pts)
        return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

    @classmethod
    def apertura_boca(cls, pts):
        i, d, s, inf = cls.BOCA
        return float(np.linalg.norm(pts[s] - pts[inf]) / (np.linalg.norm(pts[i] - pts[d]) + 1e-6))


def alinear(img, pts):
    a, b = Landmarks.ojos(pts)
    centro = (float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2))
    M = cv2.getRotationMatrix2D(centro, Landmarks.angulo(pts), 1.0)
    H, W = img.shape[:2]
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), M


def transformar_caja(caja, M):
    x, y, w, h = caja
    cx, cy = M @ np.array([x + w / 2, y + h / 2, 1.0])
    return int(cx - w / 2), int(cy - h / 2), w, h

def a_bgra(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def cargar_imagenes(carpeta):
    imgs = {}
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        print("[aviso] No existe", carpeta)
        return imgs
    for f in carpeta.iterdir():
        if f.suffix.lower() in EXTENSIONES:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[f.stem.lower()] = a_bgra(img)
    return imgs


def rotar_bgra(img, angulo):
    if abs(angulo) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def overlay_bgra(frame, img, x, y):
    h, w = img.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 >= x2 or y1 >= y2:
        return frame
    trozo = img[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = trozo[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (alpha * trozo[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    return frame


def sticker(frame, img, centro, ancho, angulo=0.0):
    h, w = img.shape[:2]
    ancho = max(int(ancho), 4)
    alto = max(int(ancho * h / w), 4)
    m = rotar_bgra(cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_AREA), angulo)
    mh, mw = m.shape[:2]
    return overlay_bgra(frame, m, int(centro[0] - mw / 2), int(centro[1] - mh / 2))


def texto(img, s, x, y, tam=0.55, color=(255, 255, 255), grosor=2):
    cv2.putText(img, s, (x, y), FONT, tam, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, tam, color, grosor, cv2.LINE_AA)

def fijar_exposicion(cap, exposicion):
    if exposicion is None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposicion)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)


class Camara:
    def __init__(self, idx=0, w=640, h=480, exposicion=None):
        self.cap = None
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            c = cv2.VideoCapture(idx, backend)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            t0, ok = time.time(), False
            while time.time() - t0 < 3:
                ok, _ = c.read()
                if ok:
                    break
            if ok:
                self.cap = c
                fijar_exposicion(c, exposicion)
                break
            c.release()
        if self.cap is None:
            raise RuntimeError("No llega ningun frame (camara ocupada u otro indice)")
        self.frame, self.fid, self.run = None, 0, True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.fid = f, self.fid + 1

    def read(self):
        """(id_frame, copia_del_frame). El id permite saber si el frame es nuevo."""
        with self.lock:
            return self.fid, (None if self.frame is None else self.frame.copy())

    def release(self):
        self.run = False
        time.sleep(0.1)
        self.cap.release()

EMOCIONES = ["angry", "duckface", "fear", "happy", "neutral", "sad", "surprise"]

TAM = 64        # tamaño del rostro para HOG y LBP
TAM_PIX = 32    # tamaño del rostro para autocaras
TAM_CNN = 48    # tamaño de entrada de la CNN

def redimensionar(gray, tam):
    """Redimensiona a tam x tam (INTER_AREA al reducir, que evita el aliasing)."""
    if gray.shape[0] == tam and gray.shape[1] == tam:
        return gray
    interp = cv2.INTER_AREA if max(gray.shape[:2]) > tam else cv2.INTER_LINEAR
    return cv2.resize(gray, (tam, tam), interpolation=interp)


def _normalizar(gray, tam):
    return cv2.equalizeHist(redimensionar(gray, tam))


def cargar_dataset(raiz, tam=None):
    imgs, y = [], []
    for idx, emo in enumerate(EMOCIONES):
        carpeta = Path(raiz) / emo
        if not carpeta.is_dir():
            print(f"[aviso] No existe {carpeta}: la clase '{emo}' no tendrá muestras")
            continue
        for f in carpeta.iterdir():
            if f.suffix.lower() not in EXTENSIONES:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if tam:
                img = redimensionar(img, tam)
            imgs.append(img)
            y.append(idx)
    y = np.array(y)
    print(f"{raiz}: {len(imgs)} imágenes | por clase:",
          {EMOCIONES[i]: int((y == i).sum()) for i in range(len(EMOCIONES))})
    return imgs, y

def feat_hog(gray):
    g = _normalizar(gray, TAM)
    return hog(g, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")


def feat_lbp(gray, p=8, r=1, celdas=(8, 8)):
    g = _normalizar(gray, TAM)
    lbp = local_binary_pattern(g, p, r, method="uniform")
    n_bins = p + 2
    ch, cw = lbp.shape[0] // celdas[0], lbp.shape[1] // celdas[1]
    feats = []
    for i in range(celdas[0]):
        for j in range(celdas[1]):
            celda = lbp[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            hist, _ = np.histogram(celda, bins=n_bins, range=(0, n_bins))
            feats.append(hist / max(hist.sum(), 1))
    return np.concatenate(feats)


def feat_hog_lbp(gray):
    return np.concatenate([feat_hog(gray), feat_lbp(gray)])


def feat_pixeles(gray):
    return _normalizar(gray, TAM_PIX).astype(np.float32).ravel() / 255.0


EXTRACTORES = {
    "hog": feat_hog,
    "lbp": feat_lbp,
    "hog_lbp": feat_hog_lbp,
    "pixeles": feat_pixeles,
}


class Extractor(BaseEstimator, TransformerMixin):

    def __init__(self, tipo="hog"):
        self.tipo = tipo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        f = EXTRACTORES[self.tipo]
        return np.array([f(img) for img in X], dtype=np.float32)

def preparar_cnn(gray):
    g = _normalizar(gray, TAM_CNN)
    return (g.astype(np.float32) / 255.0)[..., None]

def informe(yte, pred, t_entreno=None, t_pred_total=None):
    etiquetas = list(range(len(EMOCIONES)))
    print(classification_report(yte, pred, labels=etiquetas,
                                target_names=EMOCIONES, zero_division=0))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(confusion_matrix(yte, pred, labels=etiquetas))
    print(f"\nAccuracy test: {accuracy_score(yte, pred):.4f} | "
          f"F1 macro: {f1_score(yte, pred, average='macro', zero_division=0):.4f}")
    if t_entreno is not None:
        print(f"Tiempo de entrenamiento: {t_entreno:.1f} s")
    if t_pred_total is not None:
        print(f"Inferencia (solo clasificador): {1000 * t_pred_total / len(yte):.3f} ms/imagen")
    return {"accuracy": float(accuracy_score(yte, pred)),
            "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0))}

RUTA_REGISTRO = Path("informe_entrenamiento") / "registro_modelos.json"


def registrar_entrenamiento(archivo_modelo, t_entreno, **extra):
    import json
    RUTA_REGISTRO.parent.mkdir(exist_ok=True)
    datos = json.loads(RUTA_REGISTRO.read_text(encoding="utf-8")) if RUTA_REGISTRO.exists() else {}
    datos[Path(archivo_modelo).name] = {"t_entreno_s": float(t_entreno), **extra}
    RUTA_REGISTRO.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Registrado en", RUTA_REGISTRO)
