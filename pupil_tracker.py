"""
Rastreamento do movimento da pupila (íris) pela webcam.

Instalação:
    pip install opencv-python mediapipe numpy

Teclas:
    c  -> calibrar (olhe para o CENTRO da tela e aperte 'c')
    q  -> sair

Convenção de ângulo (relógio):
    0°   = cima        (12h)
    90°  = direita     (3h)
    180° = baixo       (6h)
    270° = esquerda    (9h)

Código de 3 bits (8 direções) + flag de centro:
    000 cima        (0°,   12h)
    001 cima-dir    (45°,  1:30)
    010 direita     (90°,  3h)
    011 baixo-dir   (135°, 4:30)
    100 baixo       (180°, 6h)
    101 baixo-esq   (225°, 7:30)
    110 esquerda    (270°, 9h)
    111 cima-esq    (315°, 10:30)
Quando o olhar está no centro (zona morta) o código é "centro" (flag à parte,
ou 4 bits: 1000).
"""

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------------
# PARÂMETROS AJUSTÁVEIS
# ----------------------------------------------------------------------------
GAIN = 6.0             # amplifica o deslocamento da íris (~±0.15 da largura do olho)
DEADZONE = 0.25        # raio (após ganho) abaixo do qual é "centro"
DEADZONE_EXIT = 0.32   # histerese: para SAIR do centro precisa passar disso
ANGLE_HYST = 8.0       # graus de histerese entre setores (evita oscilar na borda)
EAR_BLINK = 0.18       # abaixo disso considera piscada (ignora a amostra)

# One Euro Filter (filtro passa-baixa adaptativo em frequência)
MIN_CUTOFF = 1.5       # Hz  - menor = mais suave parado (menos tremor)
BETA = 0.8             # maior = menos atraso quando o olho se move rápido
D_CUTOFF = 1.0         # Hz  - suavização da derivada (velocidade)

# Limiares de velocidade (unidades normalizadas / s)
SPEED_SACCADE = 3.0    # acima disso = sacada (movimento rápido)

LABELS_8 = [
    "CIMA", "CIMA-DIREITA", "DIREITA", "BAIXO-DIREITA",
    "BAIXO", "BAIXO-ESQUERDA", "ESQUERDA", "CIMA-ESQUERDA",
]
CLOCK_8 = ["12h", "1:30", "3h", "4:30", "6h", "7:30", "9h", "10:30"]

# Índices MediaPipe Face Mesh (com refine_landmarks=True)
L_IRIS, R_IRIS = 468, 473
L_CORNERS = (33, 133)      # externo, interno
R_CORNERS = (362, 263)     # interno, externo
L_LIDS = (159, 145)        # superior, inferior
R_LIDS = (386, 374)


# ----------------------------------------------------------------------------
# FILTRO ONE EURO (passa-baixa com frequência de corte adaptativa)
#  - olho parado  -> corte baixo  -> remove o tremor/oscilação
#  - olho rápido  -> corte alto   -> acompanha sem atraso
# ----------------------------------------------------------------------------
class OneEuro:
    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x, 0.0
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat, dx_hat      # posição filtrada, velocidade filtrada


# ----------------------------------------------------------------------------
# GEOMETRIA DO OLHAR
# ----------------------------------------------------------------------------
def pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])


def eye_offset(lm, iris, corners, lids, w, h):
    """Deslocamento da íris em relação ao centro do olho, normalizado pela
    largura do olho. Retorna (dx, dy, EAR)."""
    c = pt(lm, iris, w, h)
    a, b = pt(lm, corners[0], w, h), pt(lm, corners[1], w, h)
    up, lo = pt(lm, lids[0], w, h), pt(lm, lids[1], w, h)
    width = np.linalg.norm(a - b) + 1e-6
    center = (a + b) / 2.0
    # centro vertical = média das pálpebras e da linha dos cantos
    center_y = (center[1] + (up[1] + lo[1]) / 2.0) / 2.0
    dx = (c[0] - center[0]) / width
    dy = (c[1] - center_y) / width
    ear = np.linalg.norm(up - lo) / width
    return dx, dy, ear


def angle_clock(dx, dy):
    """0° = cima, 90° = direita, 180° = baixo, 270° = esquerda (horário)."""
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def quadrant(dx, dy):
    if dx >= 0 and dy < 0:
        return "Q1 sup-dir"
    if dx >= 0 and dy >= 0:
        return "Q2 inf-dir"
    if dx < 0 and dy >= 0:
        return "Q3 inf-esq"
    return "Q4 sup-esq"


class GazeClassifier:
    """Classifica em 8 setores (3 bits) + centro, com histerese."""

    def __init__(self):
        self.sector = None        # None = centro
        self.in_center = True

    def update(self, gx, gy):
        r = math.hypot(gx, gy)
        # histerese radial (centro <-> fora)
        if self.in_center:
            if r > DEADZONE_EXIT:
                self.in_center = False
        else:
            if r < DEADZONE:
                self.in_center = True
        if self.in_center:
            self.sector = None
            return None, 0.0

        ang = angle_clock(gx, gy)
        new = int(round(ang / 45.0)) % 8
        if self.sector is None:
            self.sector = new
        elif new != self.sector:
            # só troca de setor se passar da borda por ANGLE_HYST graus
            center_ang = self.sector * 45.0
            diff = abs((ang - center_ang + 180) % 360 - 180)
            if diff > 22.5 + ANGLE_HYST:
                self.sector = new
        return self.sector, ang


# ----------------------------------------------------------------------------
# LOOP PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    model_path = Path(__file__).with_name("face_landmarker.task")
    face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    fx, fy = OneEuro(), OneEuro()
    clf = GazeClassifier()
    cal = np.array([0.0, 0.0])          # offset de calibração
    raw_last = None
    last_good = (0.0, 0.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)      # espelha: direita do usuário = direita da tela
        h, w = frame.shape[:2]
        t = time.monotonic()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = face_landmarker.detect_for_video(mp_image, int(t * 1000))
        info = "sem rosto"

        if res.face_landmarks:
            lm = res.face_landmarks[0]

            # após o flip, o olho "L" do MediaPipe fica do lado oposto, mas
            # a média dos dois olhos elimina essa diferença.
            ldx, ldy, lear = eye_offset(lm, L_IRIS, L_CORNERS, L_LIDS, w, h)
            rdx, rdy, rear = eye_offset(lm, R_IRIS, R_CORNERS, R_LIDS, w, h)
            ear = (lear + rear) / 2.0
            raw = np.array([(ldx + rdx) / 2.0, (ldy + rdy) / 2.0])
            raw_last = raw

            if ear < EAR_BLINK:
                gx, gy = last_good          # piscada: mantém última leitura
                blink = True
            else:
                g = (raw - cal) * GAIN
                gx, vx = fx(g[0], t)
                gy, vy = fy(g[1], t)
                last_good = (gx, gy)
                blink = False

            vx = fx.dx_prev
            vy = fy.dx_prev
            speed = math.hypot(vx, vy)

            sector, ang = clf.update(gx, gy)
            if sector is None:
                label, code, clock = "CENTRO", "----", "-"
                ang_txt = "-"
            else:
                label = LABELS_8[sector]
                code = format(sector, "03b")
                clock = CLOCK_8[sector]
                ang_txt = f"{ang:5.1f}"
            mov = "SACADA" if speed > SPEED_SACCADE else ("fixação" if speed < 0.3 else "suave")

            # desenhar íris
            for idx in (L_IRIS, R_IRIS):
                p = pt(lm, idx, w, h).astype(int)
                cv2.circle(frame, tuple(p), 3, (0, 255, 0), -1)

            info = (f"{label} | bits={code} | ang={ang_txt} | {clock} | "
                    f"{quadrant(gx, gy)} | v={speed:4.2f}/s {mov}"
                    + (" | PISCADA" if blink else ""))

            # mini-radar do olhar
            cx, cy, R = 90, h - 90, 70
            cv2.circle(frame, (cx, cy), R, (200, 200, 200), 1)
            cv2.circle(frame, (cx, cy), int(DEADZONE * R), (120, 120, 120), 1)
            cv2.line(frame, (cx - R, cy), (cx + R, cy), (120, 120, 120), 1)
            cv2.line(frame, (cx, cy - R), (cx, cy + R), (120, 120, 120), 1)
            px = int(cx + np.clip(gx, -1, 1) * R)
            py = int(cy + np.clip(gy, -1, 1) * R)
            cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)

        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2)
        cv2.putText(frame, "c=calibrar centro  q=sair", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("Pupil tracker", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c") and raw_last is not None:
            cal = raw_last.copy()
            fx, fy = OneEuro(), OneEuro()
            print("Calibrado:", cal)

    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()