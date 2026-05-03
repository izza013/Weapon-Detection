import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov10_best.pt")

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("Weapon Detection (YOLOv10)")
st.image(
    "vv.JPG",
    caption="Weapon Detection using YOLOv10",
    use_column_width=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Inference
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    results = model(img_array)

    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.image(result_img, caption="Detections", use_column_width=True)
