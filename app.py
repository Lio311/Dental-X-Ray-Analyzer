import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --- הגדרות עמוד ---
st.set_page_config(layout="wide", page_title="Dental X-Ray Analyzer")
st.title("🦷 מפענח צילומי רנטגן דנטליים")
st.markdown("העלה צילום רנטגן (נשך, פריאפיקלי או פנורמי) כדי לזהות עששת.")

# --- הגדרת נתיב המודל ---
# !!! חשוב !!!
# שנה את שם הקובץ הזה לשם קובץ המודל שאימנת
MODEL_PATH = "your_dental_model.pt"

# --- טעינת המודל ---
# שימוש ב-st.cache_resource כדי לטעון את המודל פעם אחת בלבד
@st.cache_resource
def load_model(model_path):
    """
    טוען את מודל ה-YOLOv8 מהנתיב שסופק.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {e}")
        st.info(f"ודא שקובץ בשם '{model_path}' נמצא באותה התיקייה.")
        return None

# --- פונקציות עזר ---
def draw_detections(image, results):
    """
    מצייר את תיבות הזיהוי (bounding boxes) על התמונה.
    Image: PIL Image
    results: תוצאות שהתקבלו ממודל YOLO
    """
    # המרת תמונת PIL למערך NumPy (פורמט OpenCV)
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # המרה מ-RGB ל-BGR

    # לולאה על כל התוצאות
    for result in results:
        boxes = result.boxes  # אובייקט התיבות
        names = result.names  # שמות הקלאסים
        
        for box in boxes:
            # קבלת קואורדינטות, ביטחון וקלאס
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_index = int(box.cls[0])
            label = f"{names[cls_index]} {conf:.2f}"
            
            # הגדרת צבע (אדום לעששת)
            color = (0, 0, 255) # BGR
            
            # ציור המלבן
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # ציור הטקסט (התווית)
            cv2.putText(img_cv, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # המרה חזרה מ-BGR ל-RGB עבור תצוגה ב-Streamlit
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_rgb

# --- טעינת המודל ---
model = load_model(MODEL_PATH)

# --- ממשק המשתמש (UI) ---
if model is None:
    st.warning("המודל לא נטען. האפליקציה לא יכולה לבצע זיהוי.")
else:
    st.success(f"מודל '{MODEL_PATH}' נטען בהצלחה!")

uploaded_file = st.file_uploader("בחר קובץ תמונה...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None and model is not None:
    # קריאת התמונה
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="צילום מקורי", use_container_width=True)

    with st.spinner("מבצע ניתוח..."):
        # YOLO עובד הכי טוב עם נתיב קובץ, לכן נשמור קובץ זמני
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
            tfile.write(uploaded_file.getbuffer())
            temp_path = tfile.name

        # הרצת המודל על התמונה
        # אפשר להתאים את רף הביטחון (conf)
        results = model.predict(temp_path, conf=0.35) 
        
        # ציור התוצאות
        image_with_detections = draw_detections(image, results)
        
        # מחיקת הקובץ הזמני
        os.remove(temp_path)

    with col2:
        st.image(image_with_detections, caption="צילום עם זיהוי עששת", use_container_width=True)
    
    # הצגת סיכום התוצאות
    st.subheader("סיכום ממצאים")
    detection_count = 0
    for result in results:
        detection_count += len(result.boxes)
        
    if detection_count == 0:
        st.success("לא זוהו ממצאים חשודים (מעל רף הביטחון).")
    else:
        st.warning(f"זוהו **{detection_count}** ממצאים חשודים:")
        # פירוט הממצאים
        for i, result in enumerate(results):
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                st.write(f"- ממצא: **{label}** (ביטחון: {float(box.conf[0]):.2f})")

elif uploaded_file is None:
    st.info("אנא העלה תמונה כדי להתחיל בניתוח.")
