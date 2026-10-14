import cv2
from ultralytics import YOLO

# Load a lightweight YOLO model
model = YOLO("yolov8n.pt")

# COCO class ID for cell phone
PHONE_CLASS_ID = 67

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Phone detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Detect objects in the frame
    results = model.predict(
    source=frame,
    conf=0.20,
    imgsz=640,
    verbose=False
)

    phone_detected = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Detect cell phones with confidence above 40%
            if class_id == PHONE_CLASS_ID and confidence > 0.40:
                phone_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 0, 255), 2
                )

                cv2.putText(
                    frame,
                    f"Phone: {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    if phone_detected:
        status = "WARNING: PHONE DETECTED!"
        color = (0, 0, 255)
    else:
        status = "No phone detected"
        color = (0, 255, 0)

    cv2.putText(
        frame, status, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        color, 2
    )

    cv2.imshow("DriverGaurd - Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
