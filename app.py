import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# Device setup (FIXED)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load YOLOv10 model (FIXED)
# -------------------------------
@st.cache_resource
def load_model(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        model = checkpoint

    model = model.float()
    model.to(device)
    model.eval()

    return model

# Load model
model = load_model('yolov10_best.pt')

# -------------------------------
# Preprocess image (FIXED)
# -------------------------------
def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_resized = cv2.resize(image, (640, 640))
    image_resized = np.transpose(image_resized, (2, 0, 1))
    image_resized = np.expand_dims(image_resized, axis=0)

    image_resized = torch.tensor(image_resized, dtype=torch.float32) / 255.0
    image_resized = image_resized.to(device)

    return image_resized

# -------------------------------
# Draw bounding boxes (SAFE)
# -------------------------------
def draw_boxes(image, predictions):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Handle shape safely
    if len(predictions.shape) == 3:
        predictions = predictions[0]

    for det in predictions:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, conf, cls = det

        if conf > 0.5:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls)} {conf:.2f}"

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔫 Weapon Detection using YOLOv10")

# Default image (optional)
try:
    image = Image.open('n.png')
    st.image(image, caption='Sample Image', use_column_width=True)
except:
    pass

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Inference
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        results = model(input_tensor)

        # Handle different output formats
        if isinstance(results, (list, tuple)):
            results = results[0]

        predictions = results.detach().cpu().numpy()

    image_result = draw_boxes(image, predictions)

    st.image(image_result, caption="Detections", use_column_width=True)
