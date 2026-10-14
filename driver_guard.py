import cv2
import time
import threading
import winsound
import mediapipe as mp
from ultralytics import YOLO
from math import dist
import numpy as np

# ---------------- CONFIGURATION ----------------
EAR_THRESHOLD = 0.21
CLOSED_EYE_SECONDS = 1.5
PHONE_CONFIDENCE = 0.20
ALERT_COOLDOWN = 3
# Head pose configuration
HEAD_YAW_THRESHOLD = 20       # Turn left/right threshold in degrees
HEAD_PITCH_THRESHOLD = 20     # Look up/down threshold in degrees
DISTRACTION_SECONDS = 1.5     # Duration before warning


# ---------------- LOAD MODELS ----------------
print("Loading DriverGaurd models...")

phone_model = YOLO("yolov8n.pt")

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(eye_points):
    vertical_1 = dist(eye_points[1], eye_points[5])
    vertical_2 = dist(eye_points[2], eye_points[4])
    horizontal = dist(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def estimate_head_pose(face, width, height):
    # MediaPipe face landmark indices
    landmark_ids = [1, 152, 33, 263, 61, 291]

    image_points = np.array([
        (
            face.landmark[i].x * width,
            face.landmark[i].y * height
        )
        for i in landmark_ids
    ], dtype=np.float64)

    # Approximate 3D face reference points
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    # Approximate camera parameters
    focal_length = float(width)
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return pitch, yaw, roll


# Play warning sound without freezing webcam processing
def play_alert():
    try:
        winsound.Beep(1200, 400)
    except RuntimeError:
        pass


# ---------------- START WEBCAM ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

eye_closed_since = None
last_alert_time = 0

print("DriverGaurd is running. Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # -------- DROWSINESS DETECTION --------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb_frame)

    ear = None
    drowsy = False

    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0]

        def get_eye_points(indices):
            return [
                (
                    int(face.landmark[i].x * width),
                    int(face.landmark[i].y * height)
                )
                for i in indices
            ]

        left_points = get_eye_points(LEFT_EYE)
        right_points = get_eye_points(RIGHT_EYE)

        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)
        ear = (left_ear + right_ear) / 2.0

        for point in left_points + right_points:
            cv2.circle(frame, point, 2, (255, 255, 0), -1)

        if ear < EAR_THRESHOLD:
            if eye_closed_since is None:
                eye_closed_since = time.time()

            elif time.time() - eye_closed_since >= CLOSED_EYE_SECONDS:
                drowsy = True
        else:
            eye_closed_since = None
    else:
        # Avoid counting time without a visible face as closed eyes
        eye_closed_since = None

    # -------- PHONE DETECTION --------
    phone_detected = False

    phone_results = phone_model.predict(
        source=frame,
        conf=PHONE_CONFIDENCE,
        imgsz=640,
        classes=[67],
        verbose=False
    )

    for result in phone_results:
        for box in result.boxes:
            phone_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 0, 255), 2
            )

            cv2.putText(
                frame,
                f"Phone {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # -------- STATUS DISPLAY --------
    if ear is not None:
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    if drowsy:
        cv2.putText(
            frame, "WARNING: DROWSINESS DETECTED!",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2
        )

    if phone_detected:
        cv2.putText(
            frame, "WARNING: PHONE DETECTED!",
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2
        )

    if not drowsy and not phone_detected:
        cv2.putText(
            frame, "Status: ALERT",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

    # -------- AUDIO WARNING --------
    now = time.time()

    if (drowsy or phone_detected) and now - last_alert_time >= ALERT_COOLDOWN:
        threading.Thread(target=play_alert, daemon=True).start()
        last_alert_time = now

    cv2.imshow("DriverGaurd - Safety Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
face_mesh.close()
cv2.destroyAllWindows()

print("DriverGaurd stopped.")