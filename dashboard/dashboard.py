import streamlit as st
import sqlite3
import pandas as pd
import os
import time

st.set_page_config(page_title="PPE Safety Dashboard", layout="wide")

st.title("🛡️ PPE Safety Dashboard")
st.markdown("Monitor PPE compliance and restricted zone violations in real-time.")

db_path = "data/logs/ppe_logs.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM violations ORDER BY timestamp DESC", conn)
    
    total_violations = len(df)
    unique_workers = df['worker_id'].nunique() if total_violations > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Violations", total_violations)
    col2.metric("Workers Detected in Restricted Zone", unique_workers)
    
    compliance = max(0, 100 - (total_violations * 2)) # Dummy compliance logic
    col3.metric("Compliance Rate", f"{compliance}%")
    
    st.markdown("---")
    
    col_img, col_table = st.columns([1, 1])
    
    with col_img:
        st.subheader("📷 Latest Output Frame")
        if os.path.exists("data/output/latest_frame.jpg"):
            st.image("data/output/latest_frame.jpg", use_column_width=True)
            if st.button("Refresh Feed"):
                st.rerun()
        else:
            st.info("No frame processed yet. Run the main system.")
            
    with col_table:
        st.subheader("⚠️ Recent Violations")
        if not df.empty:
            st.dataframe(df.head(15), use_container_width=True)
        else:
            st.info("No violations recorded yet.")
else:
    st.error("Database not found. Please run the application first.")