import cv2
import numpy as np

from app.config import RESTRICTED_ZONE

def point_inside_polygon(point, polygon):

    return cv2.pointPolygonTest(
        np.array(polygon, np.int32),
        point,
        False
    ) >= 0

def check_zone_violation(tracked):

    violations = []

    for obj in tracked:

        x1, y1, x2, y2 = obj["bbox"]

        center = ((x1+x2)//2, (y1+y2)//2)

        inside = point_inside_polygon(
            center,
            RESTRICTED_ZONE
        )

        if inside:
            violations.append(obj)

    return violations