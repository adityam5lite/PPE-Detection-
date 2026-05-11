import cv2
from ultralytics import YOLO

from app.detector import detect_ppe
from app.tracker import track_people
from app.zones import check_zone_violation
from app.alerts import send_alert
from app.analytics import update_analytics
from app.utils import draw_box

from app.config import VIDEO_SOURCE, MODEL_PATH

def start_system():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = detect_ppe(model, frame)

        tracked = track_people(frame, detections)

        violations = check_zone_violation(tracked)

        update_analytics(violations)

        for obj in tracked:
            class_name = {0: 'hardhat', 1: 'vest', 2: 'person'}.get(obj['class'], str(obj['class']))
            draw_box(frame, obj['bbox'], f"ID:{obj['id']} {class_name}")

        for violation in violations:
            send_alert(frame, violation)

        # To avoid OpenCV imshow errors in headless environment, we write to an output file or just display it if GUI is available.
        # But we will keep imshow as per original, and also write to output file just in case.
        cv2.imwrite("data/output/latest_frame.jpg", frame)
        # cv2.imshow("PPE AI System", frame) # Commented to prevent block/error without display

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()