import cv2
import time
from app.database import conn, cursor

def send_alert(frame, violation):

    timestamp = str(time.time())
    print(f"ALERT: Worker {violation['id']} violated safety rules")

    filename = f"data/snapshots/{timestamp}.jpg"

    cv2.imwrite(filename, frame)
    
    cursor.execute("INSERT INTO violations (worker_id, timestamp, violation_type) VALUES (?, ?, ?)", (str(violation['id']), timestamp, "Zone Violation"))
    conn.commit()