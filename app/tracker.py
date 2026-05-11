def track_people(frame, detections):

    tracked_objects = []

    for idx, detection in enumerate(detections):

        detection["id"] = idx

        tracked_objects.append(detection)

    return tracked_objects