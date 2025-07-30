import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title(" Viola-Jones Face Detection App")

# Instructions
st.markdown("""
## 📋 Instructions:
1. Upload an image with faces.
2. Adjust the detection parameters as needed.
3. Pick a color for the face detection rectangles.
4. Click **Detect Faces** to view the results.
5. Click **Save Image** to save the result to your device.
""")

# Upload image
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

# Face detection parameters
scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.01)
min_neighbors = st.slider("Min Neighbors", 1, 10, 5, 1)

# Rectangle color picker
rect_color = st.color_picker("🎨 Choose Rectangle Color", "#00FF00")
rect_bgr = tuple(int(rect_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

if uploaded_file:
    # Load and convert image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), rect_bgr, 2)

    st.image(image_np, caption=f"Detected {len(faces)} face(s)", use_column_width=True)

    # Save the image with faces
    if st.button("💾 Save Image"):
        output_dir = "detected_faces"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"face_detected_{timestamp}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        st.success(f"Image saved to {output_path}")
