import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Dental X-Ray Analyzer")
st.title("🦷 Dental X-Ray Analyzer")
st.markdown("Upload a dental X-ray image (periapical, bitewing, or panoramic) to detect potential caries.")

# --- MODEL PATH SETUP ---
# !!! IMPORTANT !!!
# Make sure your trained model file is in the same directory.
MODEL_PATH = "best.pt" 

# --- MODEL LOADING (Cached) ---
# Use st.cache_resource to load the large model file only once
@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLOv8 model from the specified path.
    """
    try:
        # The YOLO class automatically handles loading weights
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.info(f"Please ensure the trained model file named '{model_path}' is in the project directory.")
        return None

# --- HELPER FUNCTION: DRAW DETECTION BOXES ---
def draw_detections(image, results):
    """
    Draws the bounding boxes and labels onto the image.
    
    Args:
        image (PIL.Image): The original image.
        results (list): The list of results from the YOLO model.
        
    Returns:
        numpy.ndarray: The image as a NumPy array with detections drawn.
    """
    # 1. Convert PIL Image to OpenCV (NumPy array, BGR format)
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 

    detection_color = (0, 0, 255) # BGR: Red (standard for Caries)

    # 2. Loop through all results and boxes
    for result in results:
        boxes = result.boxes  
        names = result.names  
        
        for box in boxes:
            # Get coordinates, confidence, and class index
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_index = int(box.cls[0])
            
            # Create the label text
            label_name = names[cls_index] if cls_index in names else f"Class {cls_index}"
            label = f"{label_name} {conf:.2f}"
            
            # Draw the rectangle (Bounding Box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), detection_color, 2)
            
            # Draw the label text above the box
            cv2.putText(img_cv, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 2)

    # 3. Convert back from BGR to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_rgb

# --- APPLICATION START ---
model =
