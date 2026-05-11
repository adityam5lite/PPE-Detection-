import streamlit as st
import sqlite3
import pandas as pd
import os

st.set_page_config(page_title="PPE Safety Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ PPE Safety Dashboard")
st.markdown("Monitor PPE compliance and restricted zone violations in real-time.")

# ── resolve paths relative to THIS file so it works both locally & on Streamlit Cloud ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path  = os.path.join(BASE_DIR, "data", "logs", "ppe_logs.db")
frame_path = os.path.join(BASE_DIR, "data", "output", "latest_frame.jpg")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM violations ORDER BY timestamp DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()

    total_violations = len(df)
    unique_workers   = df['worker_id'].nunique() if total_violations > 0 else 0
    compliance       = max(0, 100 - (total_violations * 2))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Violations", total_violations)
    col2.metric("Workers in Restricted Zone", unique_workers)
    col3.metric("Compliance Rate", f"{compliance}%")

    st.markdown("---")

    col_img, col_table = st.columns([1, 1])

    with col_img:
        st.subheader("📷 Latest Output Frame")
        if os.path.exists(frame_path):
            st.image(frame_path, use_column_width=True)
            if st.button("🔄 Refresh Feed"):
                st.rerun()
        else:
            st.info("No frame processed yet. Run the main system locally first.")

    with col_table:
        st.subheader("⚠️ Recent Violations")
        if not df.empty:
            st.dataframe(df.head(15), use_container_width=True)
        else:
            st.info("No violations recorded yet.")
else:
    st.warning("⚠️ No database found. This is a live demo — run the AI engine locally to populate data.")
    st.markdown("""
    ### How to run locally:
    ```bash
    # 1. Activate your virtual environment
    venv\\Scripts\\activate

    # 2. Run the AI detection engine
    python run.py

    # 3. Launch the dashboard
    streamlit run dashboard/dashboard.py
    ```
    """)