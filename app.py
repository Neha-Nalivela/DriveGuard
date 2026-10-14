
import cv2

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access webcam.")
        print("Check camera permissions or try index 1.")
        return

    print("Webcam started. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                "Driver Safety AI - Webcam Test",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Driver Safety AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
