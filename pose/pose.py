import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path.home() / "pupil-tracker-models" / "pose_landmarker_lite.task"
LANDMARK_VISIBILITY = 0.4
LINE_TOLERANCE_RATIO = 0.04
BODY_CENTER_GREEN_RATIO = 0.08
BODY_CENTER_RED_RATIO = 0.5
ASYMMETRY_CALIBRATION_FRAMES = 30
DARK_LINE_MIN_LENGTH = 30
DARK_LINE_MIN_ASPECT_RATIO = 3.0

POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (11, 23), (12, 24),
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (23, 24),
]
FOOT_LANDMARKS = ((27, 31, "E"), (28, 32, "D"))


def ensure_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return

    print(f"Baixando modelo de pose em: {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo baixado com sucesso.")


def landmark_point(landmark, width, height):
    return landmark.x * width, landmark.y * height


def visible_foot_centers(landmarks, width, height):
    feet = []
    for ankle_index, toe_index, label in FOOT_LANDMARKS:
        ankle = landmarks[ankle_index]
        toe = landmarks[toe_index]
        if min(ankle.visibility, toe.visibility) < LANDMARK_VISIBILITY:
            continue

        ankle_x, ankle_y = landmark_point(ankle, width, height)
        toe_x, toe_y = landmark_point(toe, width, height)
        feet.append(((ankle_x + toe_x) / 2, (ankle_y + toe_y) / 2, label))
    return feet


def body_center_and_width(landmarks, width, height):
    pairs = ((11, 12), (23, 24))
    weighted_center = 0.0
    weighted_width = 0.0
    total_confidence = 0.0
    for left_index, right_index in pairs:
        left, right = landmarks[left_index], landmarks[right_index]
        confidence = min(left.visibility, right.visibility)
        if confidence < LANDMARK_VISIBILITY:
            continue

        left_x, _ = landmark_point(left, width, height)
        right_x, _ = landmark_point(right, width, height)
        weighted_center += ((left_x + right_x) / 2) * confidence
        weighted_width += abs(right_x - left_x) * confidence
        total_confidence += confidence

    if total_confidence == 0:
        return None

    return weighted_center / total_confidence, weighted_width / total_confidence


def deviation_color(deviation_ratio):
    progress = min(
        1.0,
        max(
            0.0,
            (deviation_ratio - BODY_CENTER_GREEN_RATIO)
            / (BODY_CENTER_RED_RATIO - BODY_CENTER_GREEN_RATIO),
        ),
    )
    if progress <= 0.5:
        amount = progress * 2
        return (0, 255, round(255 * amount))
    amount = (progress - 0.5) * 2
    return (0, round(255 * (1 - amount)), 255)


def draw_screen_center_alignment(frame, body_center, baseline_center, baseline_width):
    height, width = frame.shape[:2]
    center_x = width // 2
    corrected_center_x = body_center - (baseline_center - center_x)
    deviation = abs(corrected_center_x - center_x)
    deviation_ratio = deviation / max(baseline_width, 1)
    color = deviation_color(deviation_ratio)

    cv2.line(frame, (center_x, 0), (center_x, height - 1), color, 3)
    body_x = round(corrected_center_x)
    cv2.circle(frame, (body_x, height // 2), 7, color, -1)
    cv2.line(frame, (center_x, height // 2), (body_x, height // 2), color, 2)

    if deviation_ratio <= BODY_CENTER_GREEN_RATIO:
        status = "RETO"
    elif deviation_ratio >= BODY_CENTER_RED_RATIO:
        status = "FORA DO EIXO"
    else:
        status = "DESVIANDO"
    return f"{status} | desvio: {deviation:.0f}px ({deviation_ratio:.2f} larguras do corpo)", color


def project_to_line(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return None

    fraction = ((px - x1) * dx + (py - y1) * dy) / length_squared
    closest = (x1 + fraction * dx, y1 + fraction * dy)
    distance = ((px - closest[0]) ** 2 + (py - closest[1]) ** 2) ** 0.5
    return closest, distance


def detect_dark_line(frame, roi):
    if roi is None:
        return None

    x1, y1, x2, y2 = roi
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel_size = max(3, min(region.shape[:2]) // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    candidates = []
    for contour in contours:
        if len(contour) < 2:
            continue
        rect = cv2.minAreaRect(contour)
        side_a, side_b = rect[1]
        long_side, short_side = max(side_a, side_b), min(side_a, side_b)
        if (
            long_side < DARK_LINE_MIN_LENGTH
            or short_side == 0
            or long_side / short_side < DARK_LINE_MIN_ASPECT_RATIO
        ):
            continue
        candidates.append((cv2.contourArea(contour), contour))

    if not candidates:
        return None

    _, contour = max(candidates, key=lambda candidate: candidate[0])
    vx, vy, center_x, center_y = cv2.fitLine(
        contour, cv2.DIST_L2, 0, 0.01, 0.01
    ).flatten()
    points = contour.reshape(-1, 2).astype(float)
    projections = (points[:, 0] - center_x) * vx + (points[:, 1] - center_y) * vy
    start_projection, end_projection = projections.min(), projections.max()
    start = (
        int(x1 + center_x + vx * start_projection),
        int(y1 + center_y + vy * start_projection),
    )
    end = (
        int(x1 + center_x + vx * end_projection),
        int(y1 + center_y + vy * end_projection),
    )
    return start, end


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]
    points = [landmark_point(item, width, height) for item in landmarks]

    for start, end in POSE_CONNECTIONS:
        p1, p2 = points[start], points[end]
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)


def draw_reference_line(frame, line_points, color=(255, 200, 0)):
    if len(line_points) != 2:
        return

    height, width = frame.shape[:2]
    start, end = line_points
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = (dx * dx + dy * dy) ** 0.5
    if distance == 0:
        return

    dx, dy = dx / distance, dy / distance
    length = max(width, height) * 2
    extended_start = (int(start[0] - dx * length), int(start[1] - dy * length))
    extended_end = (int(end[0] + dx * length), int(end[1] + dy * length))
    cv2.line(frame, extended_start, extended_end, color, 2)
    cv2.circle(frame, start, 5, color, -1)
    cv2.circle(frame, end, 5, color, -1)


def draw_roi(frame, roi):
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)


def draw_line_check(frame, feet, line_points):
    if len(line_points) != 2:
        return "Marque a linha com 2 cliques"

    tolerance = frame.shape[0] * LINE_TOLERANCE_RATIO
    distances = []
    for x, y, label in feet:
        projection = project_to_line((x, y), line_points[0], line_points[1])
        if projection is None:
            return "Linha invalida: marque dois pontos diferentes"

        closest, distance = projection
        on_line = distance <= tolerance
        color = (0, 220, 0) if on_line else (0, 0, 255)
        cv2.line(frame, (int(x), int(y)), (int(closest[0]), int(closest[1])), color, 2)
        cv2.circle(frame, (int(x), int(y)), 7, color, -1)
        cv2.putText(
            frame,
            f"{label}: {distance:.0f}px",
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        distances.append(distance)

    if not distances:
        return "Pes nao visiveis"
    if all(distance <= tolerance for distance in distances):
        return f"NA LINHA (tolerancia: {tolerance:.0f}px)"
    return f"FORA DA LINHA (tolerancia: {tolerance:.0f}px)"


def main():
    ensure_model()

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        landmarker.close()
        print("Não foi possível abrir a webcam.")
        return

    line_points = []
    line_mode = "follow"
    line_roi = None
    selection_start = None
    selection_end = None
    selecting_roi = False

    def handle_mouse(event, x, y, _flags, _parameter):
        nonlocal selection_start, selection_end, line_roi, selecting_roi
        if line_mode != "physical":
            if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
                line_points.append((x, y))
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            selection_start = (x, y)
            selection_end = (x, y)
            selecting_roi = True
        elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
            selection_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and selecting_roi:
            selection_end = (x, y)
            x1, x2 = sorted((selection_start[0], selection_end[0]))
            y1, y2 = sorted((selection_start[1], selection_end[1]))
            if x2 - x1 >= DARK_LINE_MIN_LENGTH and y2 - y1 >= DARK_LINE_MIN_LENGTH:
                line_roi = (x1, y1, x2, y2)
            selecting_roi = False

    asymmetry_samples = []
    baseline_center = None
    baseline_width = None
    window_name = "Centro da tela - Q sair | A calibrar | V virtual | F fisica | R recalibrar"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse)
    print(
        "Fique centralizado e parado para calibrar a assimetria do corpo. "
        "A/R: calibrar novamente; "
        "V: linha virtual (clique em dois pontos); "
        "F: linha física preta (arraste para selecionar a região do chão); "
        "R: recalibrar; Q: sair."
    )

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Não foi possível ler o frame da câmera.")
                break

            display_roi = line_roi
            if selecting_roi and selection_start and selection_end:
                x1, x2 = sorted((selection_start[0], selection_end[0]))
                y1, y2 = sorted((selection_start[1], selection_end[1]))
                display_roi = (x1, y1, x2, y2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = cv2.getTickCount() * 1000 // cv2.getTickFrequency()
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

            feet = []
            body_measurement = None
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_pose(frame, landmarks)
                feet = visible_foot_centers(landmarks, frame.shape[1], frame.shape[0])
                body_measurement = body_center_and_width(
                    landmarks, frame.shape[1], frame.shape[0]
                )

            if line_mode == "follow":
                if baseline_center is None and body_measurement is not None:
                    asymmetry_samples.append(body_measurement)
                    if len(asymmetry_samples) >= ASYMMETRY_CALIBRATION_FRAMES:
                        baseline_center = sorted(
                            sample[0] for sample in asymmetry_samples
                        )[len(asymmetry_samples) // 2]
                        baseline_width = sorted(
                            sample[1] for sample in asymmetry_samples
                        )[len(asymmetry_samples) // 2]
                active_line_points = []
                path_line_points = None
            elif line_mode == "virtual":
                active_line_points = line_points
                path_line_points = None
            else:
                active_line_points = detect_dark_line(frame, line_roi)
                path_line_points = None

            status_color = (0, 0, 255)
            if line_mode == "follow":
                center_x = frame.shape[1] // 2
                if baseline_center is None:
                    status = (
                        f"Calibrando assimetria: {len(asymmetry_samples)}/"
                        f"{ASYMMETRY_CALIBRATION_FRAMES} quadros"
                    )
                    cv2.line(
                        frame,
                        (center_x, 0),
                        (center_x, frame.shape[0] - 1),
                        (0, 200, 255),
                        3,
                    )
                elif body_measurement is None:
                    status = "Aguardando ombros ou quadris visiveis"
                else:
                    status, status_color = draw_screen_center_alignment(
                        frame,
                        body_measurement[0],
                        baseline_center,
                        baseline_width,
                    )
            else:
                draw_reference_line(frame, active_line_points or [])
            draw_roi(frame, display_roi if line_mode == "physical" else None)
            if line_mode == "physical" and line_roi is None:
                status = "Arraste para selecionar a regiao do chao"
            elif line_mode == "physical" and active_line_points is None:
                status = "Linha preta nao encontrada na regiao"
            elif line_mode != "follow":
                status = draw_line_check(frame, feet, active_line_points or [])
                status_color = (
                    (0, 220, 0)
                    if status.startswith("NA LINHA")
                    else (0, 0, 255)
                )
            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                line_mode = "follow"
                line_points = []
                asymmetry_samples.clear()
                baseline_center = None
                baseline_width = None
            if key == ord("v"):
                line_mode = "virtual"
                line_points = []
            if key == ord("f"):
                line_mode = "physical"
                line_roi = None
                line_points = []
            if key == ord("r"):
                line_points = []
                line_roi = None
                selection_start = None
                selection_end = None
                asymmetry_samples.clear()
                baseline_center = None
                baseline_width = None
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
