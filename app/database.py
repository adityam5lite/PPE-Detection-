import sqlite3

conn = sqlite3.connect(
    "data/logs/ppe_logs.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS violations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT,
    timestamp TEXT,
    violation_type TEXT
)
""")

conn.commit()