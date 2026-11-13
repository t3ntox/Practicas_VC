import csv
import json
import math
import re
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import os
from collections import defaultdict, deque



# CONSTANTES
GT_KEYS = ("lpn", "text", "plate", "label", "number", "plate_number")

CONS = "BCDFGHJKLMNPRSTVWXYZ"
PATRONES_MATRICULA = [re.compile(p) for p in (
    rf"\d{{4}}[{CONS}]{{3}}",              # España actual
    r"[A-Z]{1,2}\d{4}[A-Z]{1,2}",          # España antigua
    r"[A-Z]{2}\d{3}[A-Z]{2}",              # Francia / Italia
    r"[A-Z]{2}\d{2}[A-Z]{3}",              # Reino Unido
    r"[A-Z]{1,3}[A-Z]{1,2}\d{1,4}[EH]?",   # Alemania
    r"\d{2}[A-Z]{2}\d{2}",                 # Portugal
    r"[A-Z]{2}\d{2}[A-Z]{2}",              # Portugal
    r"\d{2}\d{2}[A-Z]{2}",                 # Portugal
    r"[1-9][A-Z]{3}\d{3}",                 # Bélgica
    r"\d{2}[A-Z]{3}\d",                    # Países Bajos
    r"\d[A-Z]{3}\d{2}",
    r"[A-Z]{2}\d{3}[A-Z]",
    r"\d{3}[A-Z]{2}\d",
    r"[A-Z]\d{3}[A-Z]{2}",
    r"[A-Z]{3}\d{2}[A-Z]",
    r"[A-Z]{3}\d{3}",                      # Polonia / otros
    r"[A-Z]{2}\d{5}",
)]

# Parámetros de fusión de tracklets
FAMILIA = {0: 'p', 3: 'm', 2: 'v', 5: 'v', 7: 'v'}
PARAM_FUSION = {'p': (5, 0.5, 0.01), 'v': (5, 0.5, 0.01), 'm': (10, 0.8, 0.02)}
GAP_ESTATICO_SEG = 120
MIN_FRAMES_ESTATICO = 15
IOU_ESTATICO = 0.5
MAYORIA_PESADOS = 0.65

# Corrección por posición
CONS_ES = "BCDFGHJKLMNPRSTVWXYZ"
LETRA_A_DIGITO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                  "S": "5", "G": "6", "T": "7", "B": "8"}
DIGITO_A_LETRA = {"0": "D", "1": "L", "2": "Z", "5": "S", "6": "G", "7": "T", "8": "B"}

PROMPT_PLACA = "OCR:"



# TEXTO
def norm(s: str) -> str:
    """Normaliza texto a mayúsculas alfanuméricas."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def lev(a: str, b: str) -> int:
    """Distancia de Levenshtein entre dos cadenas."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def matricula_valida(texto):
    """True si el texto coincide con alguna sintaxis de matrícula europea."""
    return any(p.fullmatch(texto) for p in PATRONES_MATRICULA)


def corregir(texto):
    """Corrige una lectura de 7 caracteres al formato español actual según la posición."""
    if len(texto) != 7:
        return texto
    corr = "".join(LETRA_A_DIGITO.get(ch, ch) if k < 4 else DIGITO_A_LETRA.get(ch, ch)
                   for k, ch in enumerate(texto))
    return corr if re.fullmatch(rf"\d{{4}}[{CONS_ES}]{{3}}", corr) else texto



# GEOMETRÍA
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def centro(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def diag(b):
    return math.hypot(b[2] - b[0], b[3] - b[1])


def recortar(img, box, pad=0.05, margen_px=None):
    """Recorta una región con margen porcentual o en píxeles, ajustada a la imagen."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    if margen_px is not None:
        return img[max(0, int(y1 - margen_px)):min(h, int(y2 + margen_px)),
                   max(0, int(x1 - margen_px)):min(w, int(x2 + margen_px))]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    x2 = min(w, int(x2 + pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    y2 = min(h, int(y2 + pad * bh))
    return img[y1:y2, x1:x2]



# DATASET Y MÉTRICAS
def cargar_gt(img_path, w, h, anotaciones_dir, gt_keys=GT_KEYS):
    """Carga cajas (xyxy en píxeles) y textos anotados desde el JSON de la imagen."""
    ruta = Path(anotaciones_dir) / f"{Path(img_path).stem}.json"
    if not ruta.exists():
        return []
    gts = []
    for obj in json.loads(ruta.read_text()):
        xc, yc, bw, bh = obj["rectangle"]
        box = ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
        txt = next((norm(str(obj[k])) for k in gt_keys if obj.get(k)), None) or None
        gts.append((box, txt))
    return gts


def resumen_metricas(nombre, rows, col="ocr_crudo"):
    """Imprime exact match, acierto por carácter (1 - min(1, lev/len)) y tiempos de lectura."""
    rows = [r for r in rows if r["gt"]]
    if not rows:
        return
    exact = np.mean([r["gt"] == r[col] for r in rows])
    acc_car = np.mean([1 - min(1, lev(r["gt"], r[col]) / max(len(r["gt"]), 1)) for r in rows])
    ms = [r["ms"] for r in rows]
    print(f"{nombre:<38} n={len(rows):<5} exact match={exact:.3f} | por carácter={acc_car:.3f} | "
          f"{np.mean(ms):.0f} ms de media ({np.median(ms):.0f} de mediana)")


def evaluar_dataset(lista_imagenes, detectar, leer_placa, cargar_gt, salida, dir_errores,
                    csv_nombre, etiqueta, iou_match=0.5, pad=0.05, gt_keys=GT_KEYS, delimiter=";"):
    """Evalúa detector + OCR sobre imágenes anotadas. Función única para todos los cuadernos."""
    from pathlib import Path
    salida = Path(salida)
    (salida / dir_errores).mkdir(parents=True, exist_ok=True)

    # La primera lectura es notablemente más lenta
    leer_cronometrado(np.zeros((30, 100, 3), np.uint8), leer_placa)

    filas = []
    n_gt = n_gt_texto = n_detectadas = 0
    falsos_positivos_geom = 0
    detecciones_en_gt_sin_texto = 0

    for img_path in lista_imagenes:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = cargar_gt(img_path, w, h)
        n_gt += len(gts)
        n_gt_texto += sum(t is not None for _, t in gts)

        preds = detectar(img)
        n_detectadas += len(preds)

        # OCR sobre las detecciones 
        for box, det_conf in preds:
            mejor_iou, gt_txt = 0.0, None
            for gbox, gtxt in gts:
                v = iou(box, gbox)
                if v > mejor_iou:
                    mejor_iou, gt_txt = v, gtxt

            if mejor_iou < iou_match:
                falsos_positivos_geom += 1
                continue
            if gt_txt is None or gt_txt == "NONE":
                detecciones_en_gt_sin_texto += 1
                continue

            rec = recortar(img, box, pad=pad)
            cruda, corregida, ms = leer_cronometrado(rec, leer_placa)
            filas.append(dict(imagen=img_path.name, origen="deteccion", gt=gt_txt,
                              ocr_crudo=cruda, ocr_corregido=corregida, ms=round(ms, 1),
                              det_conf=round(det_conf, 3), iou=round(mejor_iou, 3)))
            if gt_txt != cruda and rec.size > 0:
                cv2.imwrite(str(salida / dir_errores / f"{len(filas):05d}_gt-{gt_txt}_ocr-{cruda}.jpg"), rec)

        # Rrecorte exacto de la anotación
        for gbox, gtxt in gts:
            if gtxt is None or gtxt == "NONE":
                continue
            cruda, corregida, ms = leer_cronometrado(recortar(img, gbox, pad=pad), leer_placa)
            filas.append(dict(imagen=img_path.name, origen="gt_recorte", gt=gtxt,
                              ocr_crudo=cruda, ocr_corregido=corregida, ms=round(ms, 1),
                              det_conf=None, iou=1.0))

    # Guardar resultados en CSV
    csv_path = salida / csv_nombre
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0].keys()) if filas else ["vacio"], delimiter=delimiter)
        wr.writeheader()
        wr.writerows(filas)

    print(f"\nImágenes evaluadas: {len(lista_imagenes)} | placas anotadas: {n_gt} (con texto: {n_gt_texto}) "
          f"| detecciones {etiqueta}: {n_detectadas}")

    if n_gt_texto == 0:
        print(f"\nLas anotaciones no contienen texto de matrícula (claves probadas: {gt_keys}).")
    else:
        det = [r for r in filas if r["origen"] == "deteccion"]
        techo = [r for r in filas if r["origen"] == "gt_recorte"]
        print("\n--- RESUMEN DE EXACTITUD OCR (lectura cruda) ---")
        resumen_metricas(f"OCR sobre detecciones de {etiqueta}", det, "ocr_crudo")
        resumen_metricas("OCR sobre recorte anotado (techo)", techo, "ocr_crudo")
        print("\n--- CON CORRECCIÓN POR POSICIÓN ---")
        resumen_metricas(f"OCR + corrección, detecciones {etiqueta}", det, "ocr_corregido")
        resumen_metricas("OCR + corrección, recorte anotado", techo, "ocr_corregido")
        print(f"\nPlacas evaluadas encontradas por {etiqueta}: {len(det)}/{n_gt_texto}")
        print(f"Falsos positivos geométricos (IoU < {iou_match}): {falsos_positivos_geom}")
        print(f"Detecciones en placas anotadas sin texto (None): {detecciones_en_gt_sin_texto}")

    print(f"\nCSV: {csv_path}\nRecortes con error: {salida / dir_errores}")
    return filas

def evaluar_detector(detector, imagenes_dir, anotaciones_dir, salida_dir, leer_ocr,nombre_modelo="MODELO",
                    max_imagenes=None, iou_match=0.5, pad=0.05, limpiar_salida=False):
    imagenes_dir = Path(imagenes_dir)
    anotaciones_dir = Path(anotaciones_dir)
    salida_dir = Path(salida_dir)

    detectadas_dir = salida_dir / "detectadas"
    no_detectadas_dir = salida_dir / "no_detectadas"

    detectadas_dir.mkdir(parents=True, exist_ok=True)
    no_detectadas_dir.mkdir(parents=True, exist_ok=True)

    # Se limpian resultados anteriores en caso de así quererlo
    if limpiar_salida:
        for carpeta in [detectadas_dir, no_detectadas_dir]:
            for archivo in carpeta.iterdir():
                if archivo.is_file():
                    archivo.unlink()

    if not imagenes_dir.exists():
        raise FileNotFoundError(imagenes_dir.resolve())

    if not anotaciones_dir.exists():
        raise FileNotFoundError(anotaciones_dir.resolve())


    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    lista_imagenes = sorted(
        p for p in imagenes_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensiones
    )

    if max_imagenes is not None:
        lista_imagenes = lista_imagenes[:max_imagenes]

    if not lista_imagenes:
        raise ValueError("No se encontraron imágenes.")

    print(f"\nModelo: {nombre_modelo}")
    print(f"Imágenes: {len(lista_imagenes)}")


    filas = []
    total_gt = 0
    total_detectadas = 0
    total_no_detectadas = 0
    total_sin_anotacion = 0
    tiempos_ocr = []

    for indice, img_path in enumerate(lista_imagenes, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"No se pudo cargar: {img_path.name}")
            continue

        h, w = img.shape[:2]
        gts = cargar_gt(img_path, w, h, anotaciones_dir)
        if gts is None:
            gts = []

        gts_validos = [(box, texto) for box, texto in gts if texto is not None and texto != "NONE"]
        if not gts_validos:
            total_sin_anotacion += 1
            continue

        total_gt += len(gts_validos)


        predicciones = detector(img)
        candidatos = []
        for gi, (gt_box, gt_texto) in enumerate(gts_validos):
            for pi, (pred_box, conf) in enumerate(predicciones):
                valor_iou = iou(pred_box, gt_box)

                if valor_iou >= iou_match:
                    candidatos.append((valor_iou, gi, pi, pred_box, conf))

        candidatos.sort(
            key=lambda x: x[0],
            reverse=True
        )
        asignaciones = {}
        predicciones_usadas = set()

        for valor_iou, gi, pi, pred_box, conf in candidatos:
            if (gi not in asignaciones) and (pi not in predicciones_usadas):
                asignaciones[gi] = (valor_iou, pred_box, conf)
                predicciones_usadas.add(pi)

        # Una única copia por imagen.
        img_marcada = img.copy()
        fallos_imagen = 0
        aciertos_imagen = 0

        for gi, (gt_box, gt_texto) in enumerate(gts_validos):
            if gi not in asignaciones:
                total_no_detectadas += 1
                fallos_imagen += 1
                filas.append({
                    "imagen": img_path.name,
                    "gt": gt_texto,
                    "ocr_crudo": "",
                    "ocr_corregido": "",
                    "iou": 0,
                    "confianza": "",
                    "tiempo_ms": "",
                    "resultado": "no_detectada"
                })
                continue

            mejor_iou, pred_box, conf = asignaciones[gi]
            total_detectadas += 1
            aciertos_imagen += 1

            recorte = recortar(img, pred_box, pad=pad)
            cruda = ""
            corregida = ""
            ms = 0
            if recorte is not None and recorte.size > 0:
                cruda, corregida, ms = leer_cronometrado(recorte, leer_ocr)
                tiempos_ocr.append(ms)

            x1, y1, x2, y2 = map(int, pred_box)
            cv2.rectangle(img_marcada, (x1, y1), (x2, y2), (0, 255, 0), 3)
            etiqueta = corregida if corregida else "Sin lectura"
            cv2.putText(img_marcada, etiqueta, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            filas.append({
                "imagen": img_path.name,
                "gt": gt_texto,
                "ocr_crudo": cruda,
                "ocr_corregido": corregida,
                "iou": round(float(mejor_iou), 3),
                "confianza": round(float(conf), 3),
                "tiempo_ms": round(float(ms), 1),
                "resultado": "detectada"
            })

        if aciertos_imagen > 0:
            cv2.imwrite(str(detectadas_dir / img_path.name), img_marcada)

        # Si falta alguna matrícula, se guarda también la fotografía original en no_detectadas.
        if fallos_imagen > 0:
            cv2.imwrite(
                str(no_detectadas_dir / img_path.name),
                img
            )
        print(
            f"[{indice}/{len(lista_imagenes)}] "
            f"{img_path.name} | "
            f"GT: {len(gts_validos)} | "
            f"Detectadas: {aciertos_imagen} | "
            f"Fallos: {fallos_imagen}"
        )

    csv_salida = salida_dir / "resultados.csv"
    campos = ["imagen", "gt", "ocr_crudo", "ocr_corregido", "iou", "confianza", "tiempo_ms", "resultado"]

    with open(csv_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        writer.writeheader()
        writer.writerows(filas)

    filas_ocr = [
        r for r in filas
        if r["resultado"] == "detectada"
    ]

    def mostrar_metricas(titulo, columna):
        if not filas_ocr:
            print(f"{titulo}: sin datos")
            return

        exacto = np.mean([ norm(str(r["gt"])) == norm(str(r[columna])) for r in filas_ocr])
        por_caracter = np.mean([1 - min(1,lev(norm(str(r["gt"])), norm(str(r[columna]))) / max(len(norm(str(r["gt"]))), 1)) for r in filas_ocr])

        print(f"{titulo}: Exact match={exacto:.3f} | Por caracter={por_caracter:.3f}"
        )

    print("\n" + "=" * 60)
    print(f"RESUMEN: {nombre_modelo}")
    print("=" * 60)
    print(f"Imágenes procesadas: {len(lista_imagenes)}")
    print(f"Imágenes sin anotación: {total_sin_anotacion}")
    print(f"GT total: {total_gt}")
    print(f"Detectadas: {total_detectadas}")
    print(f"No detectadas: {total_no_detectadas}")

    if total_gt:
        print(f"Recall detección: {total_detectadas / total_gt * 100:.2f}%")

    mostrar_metricas("OCR crudo", "ocr_crudo")
    mostrar_metricas("OCR corregido", "ocr_corregido")

    if tiempos_ocr:
        print(f"Tiempo medio OCR: {np.mean(tiempos_ocr):.1f} ms")
        print(f"Mediana OCR: {np.median(tiempos_ocr):.1f} ms")

    print(f"\nResultados: {salida_dir.resolve()}")
    print(f"CSV: {csv_salida}")
    print("EVALUACIÓN FINALIZADA")
    return {
        "filas": filas,
        "total_gt": total_gt,
        "detectadas": total_detectadas,
        "no_detectadas": total_no_detectadas,
        "sin_anotacion": total_sin_anotacion,
        "tiempos_ocr": tiempos_ocr,
        "csv": csv_salida,
        "salida": salida_dir
    }



# OCR: PADDLEOCR
_ocr_paddle = None
_paddle_disponible = True


def get_ocr_paddle():
    """Instancia global de PaddleOCR (se crea solo la primera vez)."""
    global _ocr_paddle, _paddle_disponible
    if not _paddle_disponible:
        return None
    if _ocr_paddle is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_paddle = PaddleOCR(
                lang='es',
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device='cpu',
                enable_mkldnn=False,
            )
        except ImportError:
            print("AVISO: paddleocr no está instalado (pip install paddlepaddle paddleocr)")
            _paddle_disponible = False
            return None
    return _ocr_paddle


def _preparar_recorte(crop):
    """Escala mínima (alto 64 / ancho 200), desenfoque gaussiano y padding réplica."""
    h_c, w_c = crop.shape[:2]
    esc = max(max(1.0, 64.0 / h_c), max(1.0, 200.0 / w_c))
    g = cv2.resize(crop, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.copyMakeBorder(g, 10, 10, 10, 10, cv2.BORDER_REPLICATE)


def leer_placa_paddle(crop, ocr_inst=None):
    """Devuelve (texto normalizado, confianza media) usando PaddleOCR."""
    if crop is None or crop.size == 0:
        return "", 0.0
    reader = ocr_inst if ocr_inst is not None else get_ocr_paddle()
    if reader is None:
        return "", 0.0

    g = _preparar_recorte(crop)

    out = []
    if hasattr(reader, "predict"):
        for page in reader.predict(g) or []:
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_polys = page.get("rec_polys", [])
            if rec_polys is not None and len(rec_polys) == len(rec_texts):
                for t, s, p in zip(rec_texts, rec_scores, rec_polys):
                    p = np.asarray(p)
                    out.append((t, float(s), float(p[:, 0].mean()), float(p[:, 1].mean())))
            else:
                for t, s in zip(rec_texts, rec_scores):
                    out.append((t, float(s), 0.0, 0.0))
    elif hasattr(reader, "ocr"):
        for line in reader.ocr(g, cls=False) or []:
            if isinstance(line, list):
                for poly, (t, s) in line:
                    p = np.asarray(poly)
                    out.append((t, float(s), float(p[:, 0].mean()), float(p[:, 1].mean())))

    out = [r for r in out if norm(r[0])]
    if not out:
        return "", 0.0

    alto = g.shape[0]
    out.sort(key=lambda r: (round(r[3] / (alto / 2)), r[2]))
    texto = norm("".join(r[0] for r in out))
    return texto, float(np.mean([r[1] for r in out]))



# VLM (PaddleOCR-VL)
_ocr_vlm = None
_vlm_disponible = True


def get_ocr_vlm(device, vlm_id="PaddlePaddle/PaddleOCR-VL"):
    """Devuelve (modelo, processor) del VLM, cargándolo solo la primera vez.
    `device` es un torch.device o un str ("cuda"/"cpu")."""
    global _ocr_vlm, _vlm_disponible
    if not _vlm_disponible:
        return None
    if _ocr_vlm is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            dev_type = getattr(device, "type", str(device))
            processor = AutoProcessor.from_pretrained(vlm_id, trust_remote_code=True)
            modelo = AutoModelForCausalLM.from_pretrained(
                vlm_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if dev_type == "cuda" else torch.float32,
            ).to(device).eval()
            _ocr_vlm = (modelo, processor)
        except Exception as e:
            print(f"ERROR cargando PaddleOCR-VL: {e}")
            _vlm_disponible = False
            return None
    return _ocr_vlm


def leer_placa_vlm(crop, device, ocr_inst=None, prompt=PROMPT_PLACA):
    """Devuelve (texto normalizado, confianza media de los tokens) usando el VLM."""
    if crop is None or crop.size == 0:
        return "", 0.0
    reader = ocr_inst if ocr_inst is not None else get_ocr_vlm(device)
    if reader is None:
        return "", 0.0
    modelo, processor = reader

    g = _preparar_recorte(crop)
    pil = Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))

    mensajes = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        mensajes, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        salida = modelo.generate(
            **inputs, max_new_tokens=32, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    gen = salida.sequences[:, inputs["input_ids"].shape[1]:]
    respuesta = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    texto = norm(respuesta)
    if not texto or texto == "NONE":
        return "", 0.0

    try:
        logp = modelo.compute_transition_scores(
            salida.sequences, salida.scores, normalize_logits=True)[0]
        especiales = set(processor.tokenizer.all_special_ids)
        probs = [float(torch.exp(lp)) for lp, t in zip(logp, gen[0].tolist())
                 if t not in especiales]
        confianza = float(np.mean(probs)) if probs else 0.0
    except Exception:
        confianza = 1.0
    return texto, confianza


def leer_cronometrado(crop, leer_fn):
    """Devuelve (lectura cruda, lectura corregida por posición, milisegundos)."""
    if crop is None or crop.size == 0:
        return "", "", 0.0
    t0 = perf_counter()
    cruda, _ = leer_fn(crop)
    ms = (perf_counter() - t0) * 1000
    return cruda, corregir(cruda), ms



# TRACKING DE VEHÍCULOS
def prediccion_posicion(info, a, gap):
    """Estima la posición esperada del tracklet tras `gap` fotogramas perdidos."""
    (f0, c0), (f1, c1) = info[a]['hist'][0], info[a]['hist'][-1]
    if f1 == f0:
        return c1
    vx, vy = (c1[0] - c0[0]) / (f1 - f0), (c1[1] - c0[1]) / (f1 - f0)
    g = min(gap, 60)
    return (c1[0] + vx * g, c1[1] + vy * g)


def es_estatico(info, frames_visto, t):
    """True si el objeto ha permanecido prácticamente inmóvil."""
    if frames_visto[t] < MIN_FRAMES_ESTATICO:
        return False
    mov = math.dist(centro(info[t]['caja_ini']), centro(info[t]['caja_fin']))
    return mov / max(diag(info[t]['caja_fin']), 1) < 0.15


def fusionar_tracklets(info, frames_visto, clase_prelim, fps):
    """Une tracklets fragmentados por oclusión (vía estática o dinámica). Devuelve {tid: raíz}."""
    ids = sorted(info, key=lambda t: info[t]['ini'])
    raiz = {t: t for t in ids}
    con_sucesor = set()
    for b in ids:
        fam = FAMILIA[clase_prelim[b]]
        seg, base, por_frame = PARAM_FUSION[fam]
        b_estatico_ok = frames_visto[b] < MIN_FRAMES_ESTATICO or es_estatico(info, frames_visto, b)
        mejor, mejor_d = None, 1e9
        for a in ids:
            if a in con_sucesor or info[a]['fin'] >= info[b]['ini']:
                continue
            if FAMILIA[clase_prelim[a]] != fam:
                continue
            gap = info[b]['ini'] - info[a]['fin']
            da, db = diag(info[a]['caja_fin']), diag(info[b]['caja_ini'])

            if (fam != 'p' and gap <= GAP_ESTATICO_SEG * fps and b_estatico_ok
                    and es_estatico(info, frames_visto, a) and 0.7 <= db / max(da, 1) <= 1.4):
                s = iou(info[a]['caja_fin'], info[b]['caja_ini'])
                if s >= IOU_ESTATICO:
                    d = 0.01 * (1 - s)
                    if d < mejor_d:
                        mejor, mejor_d = a, d
                    continue

            if gap > seg * fps:
                continue
            if not 0.4 <= db / max(da, 1) <= 2.5:
                continue
            cb = centro(info[b]['caja_ini'])
            d = min(math.dist(centro(info[a]['caja_fin']), cb),
                    math.dist(prediccion_posicion(info, a, gap), cb)) / max(da, 1)
            if d <= base + por_frame * gap and d < mejor_d:
                mejor, mejor_d = a, d
        if mejor is not None:
            raiz[b] = raiz[mejor]
            con_sucesor.add(mejor)
    return raiz


def decidir_clase(v):
    """Clase final por mayoría ponderada (bus/camión necesitan mayoría clara)."""
    c = max(v, key=v.get)
    if c in (5, 7) and v[c] / sum(v.values()) < MAYORIA_PESADOS:
        return 2
    return c



# RT-DETR: PREPROCESADO, DETECCIÓN Y TRACKER
def preprocess(pil_rgb, device):
    """Redimensiona a 640x640 y devuelve el tensor (1,3,640,640) en `device`."""
    im = pil_rgb.resize((640, 640), Image.BILINEAR)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None].to(device)


def letterbox(pil_rgb, device, size=640):
    """Redimensiona manteniendo la proporción y rellena con gris. Devuelve (tensor, r, px, py)."""
    w, h = pil_rgb.size
    r = size / max(w, h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(pil_rgb.resize((nw, nh), Image.BILINEAR), (px, py))
    x = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div(255)[None].to(device)
    return x, r, px, py


def run_detector(x, mdl, post, conf, valid_classes, sizes):
    """Devuelve una lista (boxes, scores, labels) por imagen del batch."""
    with torch.no_grad():
        labels, boxes, scores = post(mdl(x), sizes)
    labels, boxes, scores = labels.cpu().numpy(), boxes.cpu().numpy(), scores.cpu().numpy()
    results = []
    for l, b, s in zip(labels, boxes, scores):
        keep = (s >= conf) & np.isin(l, np.array(valid_classes))
        results.append((b[keep], s[keep], l[keep]))
    return results


def crear_detector_rtdetr(model_plate, postprocessor_plate, device, conf, plate_classes=(0,)):
    """Devuelve `detectar(img_bgr) -> [(caja_xyxy en píxeles originales, confianza)]`
    para usar con `evaluar_dataset`. Aplica letterbox (no deforma la imagen) y deshace
    el letterbox para devolver las cajas en la imagen original."""
    tam = torch.tensor([[640, 640]], device=device)   # (w, h) del lienzo con letterbox

    def detectar(img):
        h, w = img.shape[:2]
        x, r, px, py = letterbox(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), device)
        cajas, scores, _ = run_detector(x, model_plate, postprocessor_plate,
                                        conf, list(plate_classes), tam)[0]
        preds = []
        for (a, b, c, d), s in zip(cajas, scores):
            x1, x2 = (a - px) / r, (c - px) / r
            y1, y2 = (b - py) / r, (d - py) / r
            preds.append(([max(0, min(w, x1)), max(0, min(h, y1)),
                           max(0, min(w, x2)), max(0, min(h, y2))], float(s)))
        return preds

    return detectar


def make_tracker(fps, buffer_seconds, min_conf=0.7):
    """Crea un ByteTrack de `supervision` con la memoria indicada (en segundos)."""
    import supervision as sv
    return sv.ByteTrack(
        track_activation_threshold=min_conf,
        lost_track_buffer=int(fps * buffer_seconds),
        minimum_matching_threshold=0.7,
        frame_rate=int(fps),
        minimum_consecutive_frames=4,
    )


def detect_plates_in_crops(rgb, veh_xyxy, veh_tids, conf, model_plate, postprocessor_plate,
                           device, plate_classes=(0,), max_crops=8, min_crop_side=48,
                           crop_margin=0.05):
    """Devuelve {tracker_id_vehiculo: (score, caja_en_frame)} con la mejor matrícula de cada vehículo."""
    H, W = rgb.shape[:2]
    if len(veh_xyxy) == 0:
        return {}

    areas = (veh_xyxy[:, 2] - veh_xyxy[:, 0]) * (veh_xyxy[:, 3] - veh_xyxy[:, 1])
    order = np.argsort(-areas)[:max_crops]          # los vehículos más grandes primero

    xs, meta = [], []
    for i in order:
        x1, y1, x2, y2 = veh_xyxy[i]
        bw, bh = x2 - x1, y2 - y1
        if bw < min_crop_side or bh < min_crop_side:
            continue
        mx, my = bw * crop_margin, bh * crop_margin
        cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        cx2, cy2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        x, r, px, py = letterbox(Image.fromarray(rgb[cy1:cy2, cx1:cx2]), device)
        xs.append(x)
        meta.append((int(veh_tids[i]), cx1, cy1, r, px, py, (x1, y1, x2, y2)))

    if not xs:
        return {}

    sizes = torch.tensor([[640, 640]] * len(xs), device=device)
    results = run_detector(torch.cat(xs), model_plate, postprocessor_plate,
                           conf, list(plate_classes), sizes)

    best = {}
    for (b, s, _), (tid, ox, oy, r, px, py, vb) in zip(results, meta):
        for bb, ss in zip(b, s):
            # deshacer letterbox y volver a coordenadas del frame
            box = np.array([(bb[0] - px) / r + ox, (bb[1] - py) / r + oy,
                            (bb[2] - px) / r + ox, (bb[3] - py) / r + oy])
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (vb[0] <= cx <= vb[2] and vb[1] <= cy <= vb[3]):
                continue   # matrícula del margen que pertenece a otro coche
            if tid not in best or ss > best[tid][0]:
                best[tid] = (float(ss), box)
    return best



# PROCESAMIENTO DE VÍDEO
def procesar_video_yolo(video_path, salida_video, salida_csv, detector_placas_inst, leer_placa,
                        nombre, colores, *, conf_placa, min_frames_placa, color_placa,
                        escala_texto_placa, ocr_cada, min_ancho_ocr, min_chars_placa, ocr_score_ok,
                        ocr_max_intentos, min_frames, peso_borde, margen_borde,
                        modelo_vehiculos='yolo11l.pt', tracker_cfg="./YOLO/mi_bytetrack.yaml",
                        conf_vehiculos=0.5, iou_vehiculos=0.5, imgsz=960,
                        track_kwargs=None, predict_kwargs=None, etiqueta_ocr="ocr",
                        ocr_solo_en_placas=False, max_ancho_salida=None, mostrar_progreso=False):
    """Procesamiento en dos pasadas (YOLO + ByteTrack): inferencia/tracking/OCR y renderizado/CSV."""
    from ultralytics import YOLO

    track_kwargs = track_kwargs or {}
    predict_kwargs = predict_kwargs or {}
    model_yolo = YOLO(modelo_vehiculos)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Pasada 1: Detección, tracking y lectura OCR
    print("Iniciando Pasada 1: tracking, detección y lectura OCR en vídeo...")
    results = model_yolo.track(source=video_path, stream=True, classes=list(nombre),
                               conf=conf_vehiculos, iou=iou_vehiculos, imgsz=imgsz, persist=True,
                               tracker=tracker_cfg, **track_kwargs)

    datos = []
    placas_datos = []
    votos = defaultdict(lambda: defaultdict(float))
    frames_visto = defaultdict(int)
    info = {}
    lecturas = {}
    ultimo_ocr = {}
    intentos_ocr = defaultdict(int)

    for i, r in enumerate(results):
        frame_data = []
        if r.boxes.id is not None:
            for caja, tid, c, conf in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist(),
                                          r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                x1, y1, x2, y2 = caja
                borde = (x1 <= margen_borde or y1 <= margen_borde or
                         x2 >= w - margen_borde or y2 >= h - margen_borde)
                votos[tid][c] += conf * (peso_borde if borde else 1.0)
                frames_visto[tid] += 1
                info.setdefault(tid, {'ini': i, 'caja_ini': caja, 'hist': deque(maxlen=15)})
                info[tid]['fin'], info[tid]['caja_fin'] = i, caja
                info[tid]['hist'].append((i, ((x1 + x2) / 2, (y1 + y2) / 2)))
                frame_data.append((caja, tid, c, conf))
        datos.append(frame_data)


        placas_f = []
        if any(c != 0 for _, _, c, _ in frame_data):
            rp = detector_placas_inst.predict(r.orig_img, conf=conf_placa, imgsz=imgsz, **predict_kwargs)[0]
            for pbox, pconf in zip(rp.boxes.xyxy.int().tolist(), rp.boxes.conf.tolist()):
                cx, cy = (pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2
                cands = sorted(((b[2] - b[0]) * (b[3] - b[1]), t) for b, t, c, _ in frame_data
                               if c != 0 and b[0] <= cx <= b[2] and b[1] <= cy <= b[3])
                if not cands:
                    continue
                cands = [t for _, t in cands]
                t0 = cands[0]
                if (pbox[2] - pbox[0] >= min_ancho_ocr
                        and intentos_ocr[t0] < ocr_max_intentos
                        and lecturas.get(t0, (0.0, ""))[0] < ocr_score_ok
                        and (t0 not in ultimo_ocr or i - ultimo_ocr[t0] >= ocr_cada)):
                    ultimo_ocr[t0] = i
                    intentos_ocr[t0] += 1
                    rec_placa = recortar(r.orig_img, pbox, margen_px=4)
                    texto, oconf = leer_placa(rec_placa)
                    if len(texto) >= min_chars_placa and matricula_valida(texto):
                        score = pconf * oconf
                        if t0 not in lecturas or score > lecturas[t0][0]:
                            lecturas[t0] = (score, texto)
                placas_f.append((pbox, pconf, cands))
        placas_datos.append(placas_f)

    # Fusión de tracklets y asignación de lecturas OCR por vehículo
    raiz = fusionar_tracklets(info, frames_visto, {t: max(v, key=v.get) for t, v in votos.items()}, fps)

    votos_m = defaultdict(lambda: defaultdict(float))
    frames_m = defaultdict(int)
    for tid, v in votos.items():
        for c, x in v.items():
            votos_m[raiz[tid]][c] += x
        frames_m[raiz[tid]] += frames_visto[tid]

    clase_final = {tid: decidir_clase(v) for tid, v in votos_m.items()}

    texto_veh = {}
    for t, (score, texto) in lecturas.items():
        v = raiz[t]
        if v not in texto_veh or score > texto_veh[v][0]:
            texto_veh[v] = (score, texto)
    texto_veh = {v: texto for v, (_, texto) in texto_veh.items()}

    print(f"IDs originales: {len(info)} -> tras fusionar: {len(set(raiz.values()))}")

    # Pasada 2: Renderizado de vídeo y guardado a CSV
    print("Iniciando Pasada 2: renderizado de vídeo y escritura en CSV...")
    ids_propios = {c: {} for c in nombre}
    placas_vehiculo = set()
    frames_placa = defaultdict(int)

    cap = cv2.VideoCapture(video_path)
    if max_ancho_salida:
        # Reduce la salida al ancho máximo indicado
        esc_s = min(1.0, max_ancho_salida / w)
        w_s, h_s = int(w * esc_s) // 2 * 2, int(h * esc_s) // 2 * 2
    else:
        esc_s, w_s, h_s = 1.0, w, h
    writer = cv2.VideoWriter(salida_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_s, h_s))

    csv_file = open(salida_csv, "w", newline="", encoding="utf-8")
    csv_out = csv.writer(csv_file, delimiter=';')
    csv_out.writerow(["fotograma", "tipo_objeto", "confianza_objeto", "identificador_tracking",
                      "x1", "y1", "x2", "y2",
                      "matricula", "confianza_matricula", "mx1", "my1", "mx2", "my2",
                      "texto_matricula"])

    for i, (frame_data, placas) in enumerate(zip(datos, placas_datos)):
        ok, frame = cap.read()
        if not ok:
            break

        visibles = {}
        for caja, tid, _, conf in frame_data:
            vid = raiz[tid]
            if frames_m[vid] < min_frames:
                continue
            c = clase_final[vid]
            ids_propios[c].setdefault(vid, len(ids_propios[c]) + 1)
            visibles[vid] = (caja, c, conf)

        placa_de = {}
        for pbox, pconf, cands in placas:
            vid = next((raiz[t] for t in cands
                        if raiz[t] in visibles and visibles[raiz[t]][1] != 0), None)
            if vid is None:
                continue
            frames_placa[vid] += 1
            if frames_placa[vid] < min_frames_placa:
                continue
            placas_vehiculo.add(vid)
            if vid not in placa_de or pconf > placa_de[vid][1]:
                placa_de[vid] = (pbox, pconf)

        for vid, (caja, c, conf) in visibles.items():
            x1, y1, x2, y2 = caja
            idp = ids_propios[c][vid]
            etiqueta = f"{nombre[c]} {idp}"
            tiene_texto = vid in texto_veh
            cv2.rectangle(frame, (x1, y1), (x2, y2), colores[c], 2)
            cv2.putText(frame, etiqueta, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colores[c], 2)

            if vid in placa_de:
                (px1, py1, px2, py2), pconf = placa_de[vid]
                cv2.rectangle(frame, (px1, py1), (px2, py2), color_placa, 2)
                txt = texto_veh[vid] if tiene_texto else "?"
                fuente, esc, grosor = cv2.FONT_HERSHEY_SIMPLEX, escala_texto_placa, 1
                (tw, th), base = cv2.getTextSize(txt, fuente, esc, grosor)
                bx1 = max(0, px1)
                by2 = max(th + base + 6, py1 - 2)
                by1 = by2 - th - base - 6
                cv2.rectangle(frame, (bx1, by1), (bx1 + tw + 8, by2), color_placa, -1)
                cv2.putText(frame, txt, (bx1 + 4, by2 - base - 3),
                            fuente, esc, (0, 0, 0), grosor)
                fila_placa = [1, f"{pconf:.3f}", px1, py1, px2, py2, texto_veh.get(vid, "")]
            else:
                fila_placa = [0, "", "", "", "", "", ""]
            csv_out.writerow([i, nombre[c], f"{conf:.3f}", idp, x1, y1, x2, y2] + fila_placa)

        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        texto = "  |  ".join(f"{nombre[c]}: {len(ids_propios[c])}" for c in nombre)
        texto += f"  |  matricula: {len(placas_vehiculo)}"
        n_ocr = (sum(v in texto_veh for v in placas_vehiculo) if ocr_solo_en_placas else len(texto_veh))
        texto += f"  |  {etiqueta_ocr}: {n_ocr}"
        cv2.putText(frame, texto, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(cv2.resize(frame, (w_s, h_s)) if esc_s < 1.0 else frame)
        if mostrar_progreso and i % 100 == 0:
            print(f"Pasada 2: fotograma {i}/{len(datos)}", end="\r")

    cap.release()
    writer.release()
    csv_file.close()

    print(f"Video guardado en {os.path.abspath(salida_video)} | CSV guardado en {os.path.abspath(salida_csv)}")
    print(f"Matrículas detectadas: {len(placas_vehiculo)} | Matrículas leídas con OCR: {len(texto_veh)}")


def procesar_video_rtdetr(path_video, output_dir, model, postprocessor, model_plate, postprocessor_plate,
                          device, leer_placa, class_names, base_colors, *, plate_classes=(0,),
                          conf_vehiculos=0.5, plate_conf=0.3, hold_frames=5, max_crops=8,
                          min_crop_side=48, crop_margin=0.05, color_placa=(0, 165, 255),
                          escala_texto_placa=0.35, ocr_cada=5, min_ancho_ocr=40, min_chars_placa=4,
                          ocr_score_ok=0.5, ocr_max_intentos=6):
    """Detecta y sigue personas/vehículos con RT-DETR + ByteTrack, localiza la matrícula dentro del recorte
    de cada vehículo y la lee con `leer_placa`."""
    import supervision as sv

    base_class = list(class_names.keys())

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'resultado.mp4')
    output_csv = os.path.join(output_dir, 'resultado.csv')
    vid = cv2.VideoCapture(path_video)

    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    csv_file = open(output_csv, 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file, delimiter=';')
    csv_writer.writerow(['fotograma', 'tipo_objeto', 'confianza', 'identificador_tracking',
                         'x1', 'y1', 'x2', 'y2',
                         'matricula_en_su_caso', 'confianza_matricula', 'mx1', 'my1', 'mx2', 'my2',
                         'texto_matricula'])

    tracker_person = make_tracker(fps, 4)   
    tracker_vehicle = make_tracker(fps, 1.5)  
                              
    size_full = torch.tensor([[w, h]], device=device)

    ids_propios = {c: {} for c in class_names}
    class_scores = defaultdict(lambda: defaultdict(float)) 
    track_class = {}                                         
    next_num = {c: 1 for c in class_names}       

    plates_seen = set()      # vehiculos que han tenido matricula alguna vez
    last_plate = {}          # tid_vehiculo -> (score, caja, frame_idx)

    # Estructuras de tracking y lectura OCR
    ultimo_ocr = {}                 # tid_vehiculo -> frame_idx del último intento OCR
    intentos_ocr = defaultdict(int) # tid_vehiculo -> número de intentos OCR realizados
    lecturas = {}                   # tid_vehiculo -> (score_total, texto_matricula)

    frame_idx = 0

    while vid.isOpened():
        ret, image = vid.read()

        if not ret:
            break

        # conversion y resize UNA sola vez para el detector de vehiculos
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x_full = preprocess(Image.fromarray(rgb), device)

        boxes, scores, labels = run_detector(x_full, model, postprocessor, conf_vehiculos, base_class, size_full)[0]

        detections = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=labels.astype(int),
        )

        det_person = detections[detections.class_id == base_class[0]]
        det_veh = detections[np.isin(detections.class_id, base_class[1:])]
        det_veh = det_veh.with_nms(threshold=0.55, class_agnostic=False)  
        det_veh = det_veh.with_nms(threshold=0.65, class_agnostic=True)    

        det_person = tracker_person.update_with_detections(det_person)
        det_veh = tracker_vehicle.update_with_detections(det_veh)

       
        plate_by_tid = detect_plates_in_crops(rgb, det_veh.xyxy, det_veh.tracker_id, plate_conf,
                                              model_plate, postprocessor_plate, device,
                                              plate_classes=plate_classes, max_crops=max_crops,
                                              min_crop_side=min_crop_side, crop_margin=crop_margin)
        for tid, (sc, bx) in plate_by_tid.items():
            last_plate[tid] = (sc, bx, frame_idx)

            ancho_placa = bx[2] - bx[0]
            if (ancho_placa >= min_ancho_ocr
                    and intentos_ocr[tid] < ocr_max_intentos
                    and lecturas.get(tid, (0.0, ""))[0] < ocr_score_ok
                    and (tid not in ultimo_ocr or frame_idx - ultimo_ocr[tid] >= ocr_cada)):
                ultimo_ocr[tid] = frame_idx
                intentos_ocr[tid] += 1
                rec_placa = recortar(image, (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])), margen_px=4)
                texto, oconf = leer_placa(rec_placa)
                if len(texto) >= min_chars_placa and matricula_valida(texto):
                    score_ocr = sc * oconf
                    if tid not in lecturas or score_ocr > lecturas[tid][0]:
                        lecturas[tid] = (score_ocr, texto)

        detections = sv.Detections.merge([det_person, det_veh])

        frame_rows = [] 
        for box, score, label, tid in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
            grupo = 'person' if label == 0 else 'veh'
            key = (grupo, tid)

            # Suma el score de la clase detectada en este frame
            class_scores[key][label] += score
            cls = max(class_scores[key], key=class_scores[key].get)

            # Si la clase votada cambió, mueve el track a la nueva clase
            if track_class.get(key) != cls:
                old = track_class.get(key)
                if old is not None:
                    ids_propios[old].pop(key, None)
                ids_propios[cls][key] = next_num[cls]
                next_num[cls] += 1
                track_class[key] = cls

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(image, (x1, y1), (x2, y2), base_colors[cls], 2)
            cv2.putText(image, f" id: {ids_propios[cls][key]} {class_names[cls]}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_colors[cls], 2)

            frame_rows.append({
                'grupo': grupo, 'cls': cls, 'score': float(score), 'tid': tid,
                'id': ids_propios[cls][key], 'box': (int(x1), int(y1), int(x2), int(y2)),
                'plate': None,
            })

        for r in frame_rows:
            if r['grupo'] == 'veh' and r['tid'] in last_plate:
                sc, bx, f = last_plate[r['tid']]
                if frame_idx - f <= hold_frames:
                    pb = tuple(int(v) for v in bx)
                    txt = lecturas[r['tid']][1] if r['tid'] in lecturas else ""
                    r['plate'] = {'id': r['id'], 'score': sc, 'box': pb, 'texto': txt}
                    plates_seen.add(('veh', r['tid']))
                    # Renderizado de matrícula idéntico a YOLO (pone '?' si no hay lectura válida)
                    px1, py1, px2, py2 = pb
                    cv2.rectangle(image, (px1, py1), (px2, py2), color_placa, 2)
                    txt_display = txt if txt else "?"
                    fuente, esc, grosor = cv2.FONT_HERSHEY_SIMPLEX, escala_texto_placa, 1
                    (tw, th), base = cv2.getTextSize(txt_display, fuente, esc, grosor)
                    bx1 = max(0, px1)
                    by2 = max(th + base + 6, py1 - 2)
                    by1 = by2 - th - base - 6
                    cv2.rectangle(image, (bx1, by1), (bx1 + tw + 8, by2), color_placa, -1)
                    cv2.putText(image, txt_display, (bx1 + 4, by2 - base - 3),
                                fuente, esc, (0, 0, 0), grosor)

            p = r['plate']
            plate_cols = ([p['id'], f"{p['score']:.4f}", *p['box'], p['texto']] if p
                          else ['', '', '', '', '', '', ''])
            csv_writer.writerow([frame_idx, class_names[r['cls']], f"{r['score']:.4f}", r['id'],
                                 *r['box'], *plate_cols])

        cv2.rectangle(image, (0, h - 40), (w, h), (0, 0, 0), -1)
        texto = "  |  ".join(f"{class_names[c]}: {len(ids_propios[c])}" for c in class_names)
        texto += f"  |  plate: {len(plates_seen)}"
        texto += f"  |  ocr: {len(lecturas)}"
        cv2.putText(image, texto, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(image)
        frame_idx += 1

    vid.release()
    writer.release()
    csv_file.close()

    print(f"Video guardado en {output_path} | CSV guardado en {output_csv}")
    print(f"Matrículas detectadas: {len(plates_seen)} | Matrículas leídas con OCR: {len(lecturas)}")
    return plates_seen, lecturas
