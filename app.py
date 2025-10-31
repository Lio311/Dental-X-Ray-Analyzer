import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dental X-Ray Caries Analyzer",
    page_icon="🦷",
    layout="wide"
)

# --- MODEL PATH SETUP ---
# Since Streamlit loads directly from the repository, 
# this path must be relative to the root of your project folder on GitHub.
MODEL_PATH = "best.pt" 

# --- MODEL LOADING (Cached) ---
# Use st.cache_resource to load the large model file only once
@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLOv8 model from the specified path.
    If deploying via Streamlit Cloud, ensure 'best.pt' is in the root directory.
    """
    try:
        # YOLO class handles loading weights
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.info(f"Please check if '{model_path}' is accessible in your repository.")
        return None

# --- HELPER FUNCTION: DRAW DETECTION BOXES ---
def draw_detections(image, results):
    """
    Draws the bounding boxes and labels onto the image.
    """
    # 1. Convert PIL Image to OpenCV (NumPy array, BGR format)
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 

    detection_color = (0, 0, 255) # BGR: Red for the boxes

    # 2. Loop through all results and boxes
    for result in results:
        boxes = result.boxes  
        names = result.names  
        
        for box in boxes:
            # Get coordinates, confidence, and class index
            # .cpu().numpy() is often necessary when running on a GPU environment 
            # to move tensors to the CPU before conversion to integers.
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_index = int(box.cls[0].cpu().numpy())
            
            # Create the label text
            label_name = names.get(cls_index, f"Class {cls_index}")
            label = f"{label_name} {conf:.2f}"
            
            # Draw the rectangle (Bounding Box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), detection_color, 2)
            
            # Draw the label text above the box
            cv2.putText(img_cv, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 2)

    # 3. Convert back from BGR to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_rgb

# ==============================================================================
# 3. STREAMLIT UI AND EXECUTION
# ==============================================================================

st.title("Dental X-Ray Caries Analyzer")
st.markdown("Upload a dental X-ray image to identify potential cavities using a custom-trained YOLOv8 model.")

model = load_model(MODEL_PATH)

if model is None:
    st.error("Model not loaded. Please ensure 'best.pt' is available in the repository root.")
    st.stop()
else:
    st.success("YOLOv8 Model Loaded Successfully!")

uploaded_file = st.file_uploader("Choose an X-ray image file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original X-Ray Image", use_container_width=True)

    with st.spinner("Analyzing image for dental caries..."):
        
        # Save uploaded file to a temporary location for YOLO prediction
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
            tfile.write(uploaded_file.getbuffer())
            temp_path = tfile.name

        # Run the YOLO model prediction
        # Set a reasonable confidence threshold (e.g., 35%)
        results = model.predict(temp_path, conf=0.35) 
        
        # Draw the detection results
        image_with_detections = draw_detections(image, results)
        
        # Delete the temporary file
        os.remove(temp_path)

    with col2:
        st.image(image_with_detections, caption="Image with Caries Detection", use_container_width=True)
    
    # --- DISPLAY SUMMARY ---
    st.subheader("Analysis Summary")
    
    all_detections = []
    detection_count = 0
    
    for result in results:
        for box in result.boxes:
            detection_count += 1
            label = result.names.get(int(box.cls[0]), "Caries") # Use 'Caries' as a fallback label
            confidence = float(box.conf[0].cpu().numpy())
            all_detections.append((label, confidence))
        
    if detection_count == 0:
        st.info("No suspicious findings were detected above the 35% confidence threshold.")
    else:
        st.warning(f"**{detection_count}** suspicious findings detected:")
        # Display detailed list of findings
        for label, conf in all_detections:
            st.write(f"- Finding: **{label}** (Confidence: {conf:.2f})")

# If you need to make the model loading super robust for Streamlit Cloud deployment:
# Consider uploading 'best.pt' to Hugging Face Hub (as a model or dataset) 
# and using a custom script in @st.cache_resource to download it before loading YOLO.
