import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent
DETECTOR_FILE = PROJECT_DIR / "driver_guard.py"

st.set_page_config(
    page_title="DriverGaurd",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if "detector_process" not in st.session_state:
    st.session_state.detector_process = None

st.title("🚗 DriverGaurd")
st.subheader("Driver Drowsiness and Distraction Monitoring")

st.markdown(
    "A computer-vision prototype that monitors eye closure "
    "and detects visible mobile phones using your webcam."
)

# Check detector status
process = st.session_state.detector_process

if process is not None and process.poll() is not None:
    st.session_state.detector_process = None
    process = None

running = process is not None

# Status cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Monitoring", "Running" if running else "Stopped")

with col2:
    st.metric("Drowsiness Detection", "Available")

with col3:
    st.metric("Phone Detection", "Available")

st.divider()

st.header("Monitoring Controls")

left, right = st.columns(2)

with left:
    if st.button(
        "▶ Start Monitoring",
        type="primary",
        disabled=running,
        use_container_width=True
    ):
        if not DETECTOR_FILE.exists():
            st.error(f"Detector file not found: {DETECTOR_FILE}")
        else:
            try:
                flags = (
                    subprocess.CREATE_NEW_CONSOLE
                    if os.name == "nt"
                    else 0
                )

                process = subprocess.Popen(
                    [sys.executable, str(DETECTOR_FILE)],
                    cwd=str(PROJECT_DIR),
                    creationflags=flags
                )

                st.session_state.detector_process = process
                st.success("DriverGaurd detector launched.")
                st.rerun()

            except Exception as error:
                st.error(f"Could not start detector: {error}")

with right:
    if st.button(
        "■ Stop Monitoring",
        disabled=not running,
        use_container_width=True
    ):
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            process.kill()

        st.session_state.detector_process = None
        st.rerun()

st.divider()

st.header("Detection Features")

with st.expander("Drowsiness Detection"):
    st.write(
        "Uses MediaPipe facial landmarks and the Eye Aspect Ratio "
        "(EAR) to identify prolonged eye closure."
    )
    st.write("Configured eye-closure duration: 1.5 seconds.")

with st.expander("Mobile Phone Detection"):
    st.write(
        "Uses the pretrained YOLOv8n model to detect visible "
        "mobile phones in webcam frames."
    )
    st.write("Configured phone confidence threshold: 0.20.")

with st.expander("Alerts"):
    st.write(
        "The detector displays warnings and plays an audio beep "
        "when drowsiness or a phone is detected."
    )

st.caption(
    "Prototype only. Detection can be inaccurate and is not a "
    "replacement for safe driving practices."
)
