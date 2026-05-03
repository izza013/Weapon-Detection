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
# UI Header
# -------------------------------
st.title("Weapon Detection (YOLOv10)")
st.write("Upload an image or try the sample below ")

# -------------------------------
# Load sample image
# -------------------------------
sample_image = Image.open("n.png").convert("RGB")

# Toggle option
use_sample = st.checkbox("Use sample image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Select image source
# -------------------------------
if use_sample:
    image = sample_image
    st.image(image, caption="Sample Image", use_column_width=True)

elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

else:
    st.info(" Upload an image or select 'Use sample image'")
    st.stop()

# -------------------------------
# Inference
# -------------------------------
img_array = np.array(image)

with st.spinner(" Detecting objects..."):
    results = model(img_array)

# Plot results
result_img = results[0].plot()
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

st.image(result_img, caption="Detections", use_column_width=True)
