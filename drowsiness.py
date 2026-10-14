
import cv2
import mediapipe as mp
import math
import time

# MediaPipe face landmark detector
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for the left and right eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21
CLOSED_EYE_SECONDS = 1.5


def calculate_ear(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access webcam.")
        return

    closed_since = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        try:
            while True:
                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                result = face_mesh.process(rgb_frame)

                status = "No face detected"
                color = (0, 255, 255)

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark

                    left_ear = calculate_ear(
                        landmarks, LEFT_EYE, width, height
                    )
                    right_ear = calculate_ear(
                        landmarks, RIGHT_EYE, width, height
                    )
                    ear = (left_ear + right_ear) / 2.0

                    cv2.putText(
                        frame,
                        f"EAR: {ear:.3f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

                    if ear < EAR_THRESHOLD:
                        if closed_since is None:
                            closed_since = time.monotonic()

                        duration = time.monotonic() - closed_since

                        if duration >= CLOSED_EYE_SECONDS:
                            status = "DROWSINESS ALERT!"
                            color = (0, 0, 255)
                        else:
                            status = "Eyes closed"
                            color = (0, 165, 255)
                    else:
                        closed_since = None
                        status = "Alert - Eyes open"
                        color = (0, 255, 0)

                else:
                    closed_since = None

                cv2.putText(
                    frame,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.imshow("Driver Drowsiness Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
