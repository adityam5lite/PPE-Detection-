from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import sqlite3
import os

app = FastAPI()

DB_PATH = "data/logs/ppe_logs.db"
OUTPUT_FRAME_PATH = "data/output/latest_frame.jpg"

@app.get("/")
def home():
    return {"status": "running", "message": "PPE Detection API is active"}

@app.get("/violations")
def get_violations(limit: int = 50):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database not found")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

@app.get("/stats")
def get_stats():
    if not os.path.exists(DB_PATH):
        return {"total_violations": 0, "unique_workers": 0}
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM violations")
    total_violations = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT worker_id) FROM violations")
    unique_workers = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_violations": total_violations,
        "unique_workers": unique_workers
    }

@app.get("/latest-frame")
def get_latest_frame():
    if not os.path.exists(OUTPUT_FRAME_PATH):
        raise HTTPException(status_code=404, detail="Latest frame not found")
    return FileResponse(OUTPUT_FRAME_PATH, media_type="image/jpeg")