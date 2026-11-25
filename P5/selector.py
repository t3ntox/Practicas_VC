"""
Panel selector de imagenes generico (banderas, logos de partidos, lo que sea).

Cada elemento es un dict:
    {"nombre": "España", "imagen": <fuente>, "etiqueta": "ES"}   # etiqueta es opcional

`imagen` puede ser:
    - una ruta local          ("logos/psoe.png")
    - una URL                 ("https://flagcdn.com/w160/es.png")  -> se descarga y se cachea
    - un array de OpenCV      (np.ndarray BGR o BGRA)
    - None                    -> se dibuja un recuadro gris con la `etiqueta`
Las imagenes con transparencia (PNG de logos) se componen sobre `fondo`.
"""
import os
import hashlib
import urllib.request

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def cargar_imagen(fuente, carpeta_cache="cache_img", fondo=(255, 255, 255)):
    """Devuelve una imagen BGR (o None si falla). Acepta ruta, URL o ndarray."""
    if fuente is None:
        return None
    if isinstance(fuente, np.ndarray):
        img = fuente
    else:
        ruta = str(fuente)
        if ruta.lower().startswith(("http://", "https://")):
            os.makedirs(carpeta_cache, exist_ok=True)
            ext = os.path.splitext(ruta.split("?")[0])[1] or ".png"
            local = os.path.join(carpeta_cache, hashlib.md5(ruta.encode()).hexdigest() + ext)
            if not os.path.exists(local):
                try:
                    req = urllib.request.Request(ruta, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=8) as r, open(local, "wb") as f:
                        f.write(r.read())
                except Exception as e:
                    print("No se pudo descargar", ruta, "->", e)
                    return None
            ruta = local
        img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

    if img.ndim == 2:                                   # gris -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                               # BGRA -> BGR sobre `fondo`
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        bg = np.empty_like(img[..., :3], dtype=np.float32)
        bg[:] = fondo
        img = (img[..., :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
    return img


class SelectorImagenes:
    def __init__(self, items, n_ver=5, ancho_sel=70, ancho_nor=44, panel_w=96,
                 lado="derecha", zona_sup=88, margen_inf=104, fondo=(255, 255, 255),
                 carpeta_cache="cache_img"):
        """
        items      : lista de dicts {"nombre", "imagen", "etiqueta"?}
        n_ver      : cuantos elementos se ven a la vez (el seleccionado va en el centro)
        ancho_sel  : lado maximo (ancho y alto) de la miniatura seleccionada
        ancho_nor  : lado maximo de las miniaturas vecinas
        lado       : "derecha" o "izquierda"
        zona_sup / margen_inf : limites verticales donde se centra el panel
        """
        self.items = items
        self.n_ver, self.ancho_sel, self.ancho_nor = n_ver, ancho_sel, ancho_nor
        self.panel_w, self.lado = panel_w, lado
        self.zona_sup, self.margen_inf = zona_sup, margen_inf
        self.i = 0
        self._cache = {}
        self.imagenes = [cargar_imagen(it.get("imagen"), carpeta_cache, fondo) for it in items]
        print("Imagenes cargadas:", sum(im is not None for im in self.imagenes), "de", len(items))

    # --- navegacion -------------------------------------------------------
    def siguiente(self):
        self.i = (self.i + 1) % len(self.items)

    def anterior(self):
        self.i = (self.i - 1) % len(self.items)

    @property
    def actual(self):
        return self.items[self.i]

    @property
    def nombre(self):
        return self.items[self.i]["nombre"]

    # --- miniaturas (con cache) --------------------------------------------
    def _miniatura(self, j, lado_max):
        key = (j, lado_max)
        if key not in self._cache:
            img = self.imagenes[j]
            if img is None:      # sin imagen: recuadro gris con la etiqueta
                alto = int(lado_max * 2 / 3)
                m = np.full((alto, lado_max, 3), 90, np.uint8)
                txt = (self.items[j].get("etiqueta") or self.items[j]["nombre"][:3]).upper() or "?"
                cv2.putText(m, txt, (6, alto // 2 + 6), FONT, 0.5, (255, 255, 255), 1)
            else:                # ajusta dentro de lado_max x lado_max conservando proporcion
                h, w = img.shape[:2]
                s = min(lado_max / w, lado_max / h)
                m = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            self._cache[key] = m
        return self._cache[key]

    # --- dibujo --------------------------------------------------------------
    def dibujar(self, frame):
        H, W = frame.shape[:2]
        n = len(self.items)
        k = min(self.n_ver, n)
        idxs = [(self.i + o) % n for o in range(-(k // 2), k - k // 2)]
        elems = [(j, j == self.i, self._miniatura(j, self.ancho_sel if j == self.i else self.ancho_nor))
                 for j in idxs]
        gap = 8
        total = sum(m.shape[0] for _, _, m in elems) + gap * (len(elems) - 1) + 20
        zona_inf = H - self.margen_inf
        y = self.zona_sup + max(0, (zona_inf - self.zona_sup - total) // 2)

        if self.lado == "derecha":
            px0, px1 = W - self.panel_w, W
        else:
            px0, px1 = 0, self.panel_w
        cx = (px0 + px1) // 2

        over = frame.copy()
        cv2.rectangle(over, (px0, y - 8), (px1, y + total + 4), (20, 20, 20), -1)
        cv2.addWeighted(over, 0.55, frame, 0.45, 0, frame)

        for j, sel, m in elems:
            h, w = m.shape[:2]
            x = cx - w // 2
            frame[y:y + h, x:x + w] = m
            if sel:
                cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 255, 255), 2)
                nombre = self.items[j]["nombre"][:14]
                tw = cv2.getTextSize(nombre, FONT, 0.42, 1)[0][0]
                cv2.putText(frame, nombre, (cx - tw // 2, y + h + 16), FONT, 0.42, (255, 255, 255), 1)
                y += 20
            y += h + gap


# ---------------------------------------------------------------------------
# Ejemplos de uso
# ---------------------------------------------------------------------------
# Banderas (a partir de la lista `paises` del notebook):
#
#   items = [{"nombre": p["pais"],
#             "imagen": f"https://flagcdn.com/w160/{iso_de(p)}.png",
#             "etiqueta": iso_de(p)} for p in paises]
#   selector = SelectorImagenes(items)
#
# Logos de partidos (rutas locales o URLs):
#
#   partidos = [{"nombre": "Partido A", "imagen": "logos/a.png", "etiqueta": "A"},
#               {"nombre": "Partido B", "imagen": "logos/b.png", "etiqueta": "B"}]
#   selector = SelectorImagenes(partidos, lado="izquierda", ancho_sel=80)
#
# En el bucle principal:
#
#   selector.dibujar(frame)
#   ...
#   elif k == ord('n'): selector.siguiente(); med.reset()
#   elif k == ord('b'): selector.anterior();  med.reset()
#
# Para obtener el elemento elegido (en vez de `paises[pais_i]`):
#
#   p = paises[selector.i]        # mismo orden que `items`
#   selector.nombre               # nombre del elemento actual