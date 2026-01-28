import streamlit as st
import cv2
import numpy as np
import sys
from pathlib import Path

# Add root to sys.path to allow imports if needed, though we use relative imports here
# if running as a module/script inside the folder.
# But streamlit runs the script directly. 
# We are in streamlit/app.py. Let's make sure we can import from local utils.
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import utils_card_extraction as card_ext
import utils_content_recognition as content_rec

st.set_page_config(layout="wide", page_title="Business Card Extraction")

st.title("Business Card Information Extraction")

# =========================
# Sidebar Config
# =========================
st.sidebar.header("Model Configuration")

# Default Paths (Relative to project root, assuming running from root)
# But we need to handle paths robustly.
PROJECT_ROOT = current_dir.parent

DEFAULT_SEG_MODEL = PROJECT_ROOT / "streamlit/models/segmentation/best.pt"
DEFAULT_CLS_MODEL = PROJECT_ROOT / "streamlit/models/classification/best.pt"
DEFAULT_CONTENT_MODEL = PROJECT_ROOT / "streamlit/models/content/best.pt"

seg_model_path = st.sidebar.text_input("Segmentation Model Path", str(DEFAULT_SEG_MODEL))
cls_model_path = st.sidebar.text_input("Classification (Upright) Model Path", str(DEFAULT_CLS_MODEL))
content_model_path = st.sidebar.text_input("Content Recognition Model Path", str(DEFAULT_CONTENT_MODEL))

st.sidebar.subheader("Inference Settings")
conf_seg = st.sidebar.slider("Segmentation Confidence", 0.1, 1.0, 0.25)
conf_content = st.sidebar.slider("Content Detection Confidence", 0.1, 1.0, 0.25)
device = st.sidebar.selectbox("Device", ["0", "cpu"], index=0)

# =========================
# Model Loading
# =========================
@st.cache_resource
def load_models(seg_path, cls_path, content_path):
    s_model = card_ext.load_seg_model(seg_path)
    c_model = card_ext.load_cls_model(cls_path)
    cnt_model = content_rec.load_content_model(content_path)
    return s_model, c_model, cnt_model

@st.cache_resource
def load_ocr():
    return content_rec.load_ocr_reader(gpu=(device != "cpu"))

try:
    with st.spinner("Loading models..."):
        seg_model, cls_model, content_model = load_models(seg_model_path, cls_model_path, content_model_path)
        ocr_reader = load_ocr()
    st.sidebar.success("Models Loaded!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# =========================
# Main UI
# =========================

st.header("Step 1: Input Image")

input_method = st.radio("Input Method", ["Upload Image", "Use Example Image"], horizontal=True)

image = None
image_name = ""

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1) # BGR
        image_name = uploaded_file.name

elif input_method == "Use Example Image":
    st.write("Select an example image:")
    # Calculate example paths
    example_dir = current_dir / "examples"
    ex_images = {
        "Example 1": example_dir / "example_2cards.jpg",
        "Example 2": example_dir / "example_3cards.jpg",
        "Example 3": example_dir / "example_4cards.jpg"
    }
    
    # Columns for thumbnails
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        st.image(str(ex_images["Example 1"]), caption="Example 1", use_container_width=True)
    with col_ex2:
        st.image(str(ex_images["Example 2"]), caption="Example 2", use_container_width=True)
    with col_ex3:
        st.image(str(ex_images["Example 3"]), caption="Example 3", use_container_width=True)

    selected_example = st.radio("Choose Example:", list(ex_images.keys()), horizontal=True)
    
    if selected_example:
        ex_path = ex_images[selected_example]
        if ex_path.exists():
            image = cv2.imread(str(ex_path))
            image_name = selected_example

if image is not None:
    # Display Original
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

    
    # Step 2: Extract Cards
    st.header("Step 2: Card Extraction")
    
    if st.button("Extract Cards"):
        with st.spinner("Extracting cards..."):
            extracted_cards = card_ext.extract_card_seg(
                image, 
                seg_model, 
                cls_model, 
                conf_seg=conf_seg, 
                device=device
            )
        
        if not extracted_cards:
            st.warning("No cards detection.")
        else:
            st.success(f"Found {len(extracted_cards)} cards.")
            
            # Store extracted cards in session state if we want to persist between reruns (not strictly needed if button logic is linear)
            # But inside button scope, we continue processing.
            
            for idx, card_data in enumerate(extracted_cards):
                st.divider()
                st.subheader(f"Card #{idx+1}")
                
                col1, col2, col3 = st.columns(3)
                
                # Show Mask (on black background for visibility or overlay)
                mask_vis = cv2.cvtColor(card_data['mask'], cv2.COLOR_GRAY2RGB)
                # Overlay mask on original for visualization
                overlay = image.copy()
                overlay[card_data['mask'] > 0] = (0, 255, 0)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, overlay)
                
                with col1:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Segmentation Mask", use_container_width=True)
                
                # Show Extracted Card
                card_img_bgr = card_data['crop_img']
                with col2:
                    st.image(cv2.cvtColor(card_img_bgr, cv2.COLOR_BGR2RGB), caption=f"Extracted & Upright (Rot: {card_data['rotation_k']*90} deg)", use_container_width=True)
                
                # Step 3: Content Recognition
                with col3:
                    st.write("Running Content Recognition...")
                    annotated_card, detections = content_rec.recognize_content(
                        card_img_bgr,
                        content_model,
                        ocr_reader,
                        conf_thres=conf_content,
                        device=device
                    )
                    st.image(cv2.cvtColor(annotated_card, cv2.COLOR_BGR2RGB), caption="Content Detection", use_container_width=True)
                
                # Show Results Table
                if detections:
                    st.write("Extracted Information:")
                    det_data = [{"Class": d['class'], "Text": d['ocr_text'], "Confidence": d['confidence']} for d in detections]
                    st.table(det_data)
                else:
                    st.info("No content detected on this card.")
