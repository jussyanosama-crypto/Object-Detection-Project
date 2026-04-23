import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

# Page config
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered"
)

# Initialize model (cache to prevent reloading)
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "best.pt")
    if os.path.exists(model_path):
        return YOLOModel(model_path)
    return None

detector = load_model()

# Header
st.title("🎯 YOLOv11 Object Detection")
st.markdown("Upload an image or use the demo image to run object detection.")

if detector is None:
    st.error("❌ Model file not found. Please ensure `model/best.pt` exists.")
    st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Main content
col1, col2 = st.columns(2)

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

use_demo = st.button("Use Demo Image")

image_to_process = None

if uploaded_file is not None:
    try:
        # Read uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        image_to_process = np.array(image)
        # Convert RGB to BGR for OpenCV processing internally
        image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
elif use_demo:
    demo_path = os.path.join("assets", "demo.png")
    if os.path.exists(demo_path):
        image_to_process = cv2.imread(demo_path)
    else:
        st.error("❌ Demo image not found at `assets/demo.png`")

if image_to_process is not None:
    # Display original
    with col1:
        st.subheader("Original Image")
        # Convert BGR back to RGB for display
        display_img = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
        st.image(display_img, use_container_width=True)
    
    # Run Detection button
    if st.button("🚀 Run Detection", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Run inference
                detections = detector.predict(image_to_process, conf=confidence_threshold)
                
                # Draw boxes
                annotated_img = draw_boxes(image_to_process.copy(), detections)
                
                # Display result
                with col2:
                    st.subheader("Detection Result")
                    display_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(display_annotated, use_container_width=True)
                
                # Display metrics/results
                if detections:
                    st.success(f"Found {len(detections)} objects!")
                    with st.expander("Show Details"):
                        for d in detections:
                            st.write(f"- **{d['class_name']}** (Confidence: {d['confidence']:.2f})")
                else:
                    st.info("No objects detected above the confidence threshold.")
            except Exception as e:
                st.error(f"Error during detection: {e}")
