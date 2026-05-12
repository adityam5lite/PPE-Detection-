import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Safety Monitor",
    page_icon="🦺",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 10px; }
    .violation-box {
        background: #2d1b1b; border: 1px solid #ff4b4b;
        border-radius: 8px; padding: 8px 12px; margin: 4px 0;
        font-size: 14px; color: #ff4b4b;
    }
    .safe-box {
        background: #1b2d1b; border: 1px solid #21c354;
        border-radius: 8px; padding: 8px 12px; margin: 4px 0;
        font-size: 14px; color: #21c354;
    }
    h1 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.title("🦺 PPE Detection & Safety Monitoring System")
st.markdown("**Real AI model** trained on construction site PPE data — detects Hardhats, Masks, Vests and flags violations in real-time.")
st.markdown("---")

# ─── Model Path ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ppe_yolo.pt")

# PPE Classes from our trained model (css-data):
#   0: Hardhat       → SAFE
#   1: Mask          → SAFE
#   2: NO-Hardhat    → VIOLATION ⚠
#   3: NO-Mask       → VIOLATION ⚠
#   4: Safety Vest   → SAFE

VIOLATION_CLASSES = {2: "NO-Hardhat ⛑", 3: "NO-Mask 😷"}
SAFE_CLASSES      = {0: "Hardhat ✅",   1: "Mask ✅",   4: "Safety Vest ✅"}

# Color map: BGR format for OpenCV
CLASS_COLORS = {
    0: (0, 200, 0),      # Hardhat      → Green
    1: (0, 200, 100),    # Mask         → Teal
    2: (0, 0, 255),      # NO-Hardhat   → Red
    3: (0, 80, 255),     # NO-Mask      → Orange-Red
    4: (255, 180, 0),    # Safety Vest  → Blue-Yellow
}

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.35, 0.05)
    skip_frames = st.slider("Process every N frames", 1, 5, 2)
    st.markdown("---")
    st.markdown("**PPE Classes Detected:**")
    st.success("✅ Hardhat")
    st.success("✅ Mask")
    st.success("✅ Safety Vest")
    st.error("🚨 NO-Hardhat")
    st.error("🚨 NO-Mask")
    st.markdown("---")
    st.info("Model: YOLOv8n trained on CSS PPE Dataset (20 epochs)")

# ─── Main Upload Section ─────────────────────────────────────────────────────
uploaded_video = st.file_uploader(
    "📁 Upload a construction site / workplace video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()

    st.video(tfile.name)
    st.success(f"✅ Video uploaded: `{uploaded_video.name}`")
    st.markdown("---")

    # ─── Load Model ──────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at `{MODEL_PATH}`. Please ensure `models/ppe_yolo.pt` exists.")
        st.stop()

    with st.spinner("🔄 Loading PPE Detection Model..."):
        model = YOLO(MODEL_PATH)
    st.success("✅ PPE Model Loaded Successfully!")

    # ─── Layout ─────────────────────────────────────────────────────────────
    col_feed, col_stats = st.columns([2, 1])

    with col_feed:
        st.subheader("📹 Live Detection Feed")
        stframe = st.empty()

    with col_stats:
        st.subheader("📊 Live Stats")
        metric_violations = st.empty()
        metric_safe        = st.empty()
        metric_frames      = st.empty()
        st.markdown("---")
        st.subheader("⚠️ Violation Log")
        violation_log      = st.empty()

    # ─── Video Processing ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(tfile.name)

    total_violations  = 0
    total_safe        = 0
    frames_processed  = 0
    frame_idx         = 0
    violation_summary = defaultdict(int)
    recent_violations = []

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed
        if frame_idx % skip_frames != 0:
            continue

        frames_processed += 1

        # ── Run PPE Detection ────────────────────────────────────────────────
        results = model(frame, conf=confidence_threshold, verbose=False)
        detections = results[0].boxes

        # ── Annotate frame manually for better control ───────────────────────
        annotated = frame.copy()
        frame_violations = 0
        frame_safe        = 0

        for box in detections:
            cls        = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(cls, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label with class + confidence
            label = f"{model.names[cls]} {conf_score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Count violations vs safe ─────────────────────────────────────
            if cls in VIOLATION_CLASSES:
                frame_violations += 1
                violation_summary[model.names[cls]] += 1
                recent_violations.append(model.names[cls])
                if len(recent_violations) > 20:
                    recent_violations.pop(0)
            elif cls in SAFE_CLASSES:
                frame_safe += 1

        # ── Overlay violation banner ─────────────────────────────────────────
        if frame_violations > 0:
            total_violations += frame_violations
            banner = f"  ⚠ {frame_violations} PPE VIOLATION(S) THIS FRAME"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (0, 0, 180), -1)
            cv2.putText(annotated, banner, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            total_safe += frame_safe
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (0, 140, 0), -1)
            cv2.putText(annotated, "  ✓ ALL PPE COMPLIANT", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Update Streamlit UI ──────────────────────────────────────────────
        stframe.image(annotated, channels="BGR", use_column_width=True)

        metric_violations.metric("🚨 Total Violations", total_violations,
                                  delta=f"+{frame_violations}" if frame_violations > 0 else None,
                                  delta_color="inverse")
        metric_safe.metric("✅ Safe Detections", total_safe)
        metric_frames.metric("🎞 Frames Analyzed", frames_processed)

        # Violation log
        if recent_violations:
            log_html = "".join(
                [f'<div class="violation-box">🚨 {v}</div>' for v in reversed(recent_violations[-8:])]
            )
        else:
            log_html = '<div class="safe-box">✅ No violations yet</div>'
        violation_log.markdown(log_html, unsafe_allow_html=True)

    cap.release()

    # ─── Final Summary ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Detection Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Frames Analyzed",  frames_processed)
    c2.metric("Total PPE Violations",   total_violations)
    compliance_pct = round(100 * (1 - total_violations / max(total_violations + total_safe, 1)), 1)
    c3.metric("Compliance Rate",        f"{compliance_pct}%")

    if violation_summary:
        st.markdown("**Violation Breakdown:**")
        for vtype, count in sorted(violation_summary.items(), key=lambda x: -x[1]):
            st.error(f"🚨 **{vtype}** — detected in **{count}** instances")
    else:
        st.success("🎉 No PPE violations detected in the entire video!")

    st.success("✅ Video Processing Complete!")

else:
    # ─── Landing Page when no video uploaded ─────────────────────────────────
    st.info("👆 Upload a workplace or construction site video above to begin PPE analysis.")
    st.markdown("""
    ### How it works:
    1. 📁 **Upload** any site video (MP4, AVI, MOV)
    2. 🤖 **AI model** scans every frame for PPE items
    3. 🚨 **Violations** are flagged in real-time (no hardhat, no mask)
    4. 📊 **Live dashboard** tracks compliance rate

    ### What the model detects:
    | Class | Status |
    |---|---|
    | Hardhat | ✅ Safe |
    | Mask | ✅ Safe |
    | Safety Vest | ✅ Safe |
    | NO-Hardhat | 🚨 Violation |
    | NO-Mask | 🚨 Violation |
    """)
