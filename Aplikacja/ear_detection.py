import cv2
from ultralytics import YOLO

#Leniwe ładowanie modelu
_model = None


def get_detection_model():
    global _model
    if _model is None:
        _model = YOLO('yolov8_wykrywanie2.pt')
    return _model


def detect_and_display_ear(frame):
    model = get_detection_model()
    results = model(frame)
    display_frame = frame.copy()
    ear_found = None

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            conf = confidences[i]
            if conf > 0.9:
                x1, y1, x2, y2 = map(int, boxes[i])
                width = x2 - x1
                height = y2 - y1

                # Rysowanie bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Ucho: {conf:.2f}, Rozmiar: {width}x{height}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Sprawdzanie minimalnego rozmiaru
                if width >= 160 and height >= 210:
                    # Wycinanie ucha
                    ear_found = frame[y1:y2, x1:x2]

    return display_frame, ear_found