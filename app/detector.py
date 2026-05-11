def detect_ppe(model, frame):

    results = model(frame)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": cls,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return detections