"""
Webcam gesture detector (Desktop version).
Reconoce 8 gestos interactivos y muestra su imagen correspondiente desde 'imagenestarea2':
  1. paz          -> imagenestarea2/paz.png (dos dedos extendidos: índice y medio)
  2. ILY          -> imagenestarea2/ILY.png (pulgar, índice y meñique extendidos)
  3. pulgarArriba -> imagenestarea2/pulgararriba.png (puño cerrado con pulgar hacia arriba)
  4. saludo       -> imagenestarea2/saludo.png (mano abierta con dedos extendidos)
  5. doblePuño    -> imagenestarea2/doblepuño.png (dos manos cerradas en puño)
  6. dobleIndice  -> imagenestarea2/dobleindice.png (dos manos apuntando con el índice)
  7. bocaAbierta  -> imagenestarea2/bocabierta.png (rostro con boca abierta)
  8. manosArriba  -> imagenestarea2/manosarriba.png (dos manos alzadas abiertas)
  9. default      -> memes/pokercat.jpg (sin gesto activo)

Presiona 'q' o ESC en la ventana de la cámara para salir.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "memes"
IMAGENES = ROOT / "imagenestarea2"

GESTURE_MEMES = {
    "paz": ["paz.png"],
    "ILY": ["ILY.png"],
    "pulgarArriba": ["pulgararriba.png"],
    "saludo": ["saludo.png"],
    "doblePuño": ["doblepuño.png"],
    "dobleIndice": ["dobleindice.png"],
    "bocaAbierta": ["bocabierta.png"],
    "manosArriba": ["manosarriba.png"],
    "default": ["pokercat.jpg"],
}

STABLE_FRAMES_REQUIRED = 4
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200
MOUTH_OPEN_THRESHOLD = 0.15

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- Ayudas geométricas 3D ------------------------------------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip):
    """Calcula si un dedo está estirado verificando el ángulo entre falanges (< 45°)."""
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45


def yaw_from_transform_matrix(matrix):
    """Extrae el ángulo Yaw (giro de cabeza) de la matriz facial 4x4."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)


def classify_hand(landmarks):
    """Analiza la mano y extrae el estado de cada dedo y puntos clave."""
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
        "wrist": pts[0],
        "palmCenter": pts[9],
        "pts": pts,
    }


def is_pointing(h):
    """Solo el dedo índice extendido."""
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


def is_peace_sign(h):
    """Símbolo de la paz: dos dedos extendidos (índice y medio), anular y meñique doblados."""
    return h["indexUp"] and h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


def is_ily_sign(h):
    """Gesto ILY: pulgar, índice y meñique estirados; medio y anular doblados."""
    return h["thumbOut"] and h["indexUp"] and h["pinkyUp"] and not h["middleUp"] and not h["ringUp"]


def is_thumbs_up(h):
    """Pulgar arriba: 4 dedos doblados y la punta del pulgar verticalmente más alta."""
    if h["curledCount"] != 4:
        return False
    pts = h["pts"]
    thumb_tip_y = pts[4][1]
    # En coordenadas de imagen, menor Y representa mayor altura física
    return (
        thumb_tip_y < pts[3][1]
        and thumb_tip_y < pts[2][1]
        and thumb_tip_y < pts[5][1]
    )


def is_saludo(h):
    """Saludo / Mano abierta: los 4 dedos extendidos."""
    return h["curledCount"] == 0 or (
        h["indexUp"] and h["middleUp"] and h["ringUp"] and h["pinkyUp"]
    )


class GestureState:
    def __init__(self):
        self.last_face = None  # (mouth_center, face_width, mouth_open, yaw_deg, t)
        self.face_seen_this_frame = False
        self.last_mouth_open = 0.0
        self.last_yaw_debug = 0.0

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / (face_width or 1e-6)

            yaw_deg = 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg = yaw_from_transform_matrix(
                    face_result.facial_transformation_matrixes[0]
                )

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, now)
            self.last_mouth_open = mouth_open
            self.last_yaw_debug = yaw_deg

        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = (
            self.last_face is not None and (now - self.last_face[4] < FACE_STALE_MS)
        )
        mouth_open = self.last_face[2] if face_is_fresh else 0.0

        # Sin manos en pantalla: verificar si la boca está abierta
        if not hand_result.hand_landmarks:
            if face_is_fresh and mouth_open > MOUTH_OPEN_THRESHOLD:
                return "bocaAbierta"
            return "default"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]

        # 1. GESTOS DE 2 MANOS
        if len(hands) >= 2:
            h0, h1 = hands[0], hands[1]

            # Manos abiertas arriba
            head_top_y = (
                (self.last_face[0][1] - self.last_face[1] * 0.8)
                if face_is_fresh
                else 0.45
            )
            hands_up = (
                h0["palmCenter"][1] < head_top_y or h0["palmCenter"][1] < 0.45
            ) and (h1["palmCenter"][1] < head_top_y or h1["palmCenter"][1] < 0.45)
            if hands_up and h0["curledCount"] <= 2 and h1["curledCount"] <= 2:
                return "manosArriba"

            # Doble puño
            if (
                h0["curledCount"] == 4
                and h1["curledCount"] == 4
                and not is_thumbs_up(h0)
                and not is_thumbs_up(h1)
            ):
                return "doblePuño"

            # Doble índice
            if is_pointing(h0) and is_pointing(h1):
                return "dobleIndice"

        # 2. GESTO FACIAL: Boca abierta
        if face_is_fresh and mouth_open > MOUTH_OPEN_THRESHOLD:
            return "bocaAbierta"

        # 3. GESTOS DE 1 MANO (evalúa cualquiera de las manos visibles)
        for h in hands:
            # Paz (dos dedos extendidos: índice y medio)
            if is_peace_sign(h):
                return "paz"

            # ILY (pulgar, índice y meñique)
            if is_ily_sign(h):
                return "ILY"

            # Pulgar arriba
            if is_thumbs_up(h):
                return "pulgarArriba"

            # Saludo (mano abierta)
            if is_saludo(h):
                return "saludo"

        return "default"


def load_memes():
    """Carga las imágenes desde imagenestarea2 (o memes como respaldo)."""
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        imgs = []
        for name in files:
            p = IMAGENES / name
            if not p.exists():
                p = MEMES / name
            if not p.exists():
                raise FileNotFoundError(
                    f"No se encontró el archivo {name} ni en {IMAGENES} ni en {MEMES}"
                )
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"Error al decodificar imagen: {p}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    """Muestra información de estado en la esquina superior izquierda."""
    lines = [
        f"Gesto actual: {gesture}",
        f"Boca abierta: {state.last_mouth_open:.2f} (umbral > {MOUTH_OPEN_THRESHOLD:.2f})",
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )


def draw_landmarks(frame, hand_result):
    """Dibuja el esqueleto de las manos sobre el cuadro de video."""
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    """Redimensiona la imagen manteniendo su proporción según la altura de la cámara."""
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS / "hand_landmarker.task")
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS / "face_landmarker.task")
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara web (índice 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # Efecto espejo

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            # Antirebote: requiere estabilidad de varios frames antes de cambiar
            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                current_meme = random.choice(memes[gesture])

            if gesture != "default":
                last_non_default_at = now
            elif (
                now - last_non_default_at > DEFAULT_FALLBACK_MS
                and current_gesture != "default"
            ):
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
