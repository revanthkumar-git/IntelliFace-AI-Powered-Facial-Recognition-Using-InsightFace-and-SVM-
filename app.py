import streamlit as st
import joblib
import cv2
import numpy as np

from PIL import Image
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Face Recognition", page_icon="👤")
st.title("Face Recognition System")
st.write("Upload an image or capture a photo to recognize faces.")

# Load trained models
clf = joblib.load("face_svm.pkl")
le = joblib.load("label_encoder.pkl")

# Initialize InsightFace
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

# PASTE YOUR EXISTING recognize_faces() FUNCTION HERE
# It should return: annotated_image, total_faces

def recognize_faces(image, threshold=0.60, min_gap=0.15):
    faces = app.get(image)
    total_faces = len(faces)
    annotated = image.copy()

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        embedding = face.normed_embedding.reshape(1, -1)

        probabilities = clf.predict_proba(embedding)[0]
        order = np.argsort(probabilities)[::-1]

        best_idx = order[0]
        best_prob = probabilities[best_idx]
        second_prob = probabilities[order[1]] if len(order) > 1 else 0.0
        gap = best_prob - second_prob

        if best_prob >= threshold and gap >= min_gap:
            prediction = clf.classes_[best_idx]

            # Handle string labels or LabelEncoder integer labels
            if isinstance(prediction, (int, np.integer)):
                name = le.inverse_transform([prediction])[0]
            else:
                name = str(prediction)

            label = f"{name} ({best_prob:.0%})"
            color = (0, 200, 0)
        else:
            label = f"Unknown (top match: {best_prob:.0%})"
            color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )

    return annotated, total_faces

def process_image(pil_image):
    image = cv2.cvtColor(
        np.array(pil_image.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )

    annotated, total = recognize_faces(image)

    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption=f"Faces detected: {total}",
        use_container_width=True
    )


upload_tab, camera_tab = st.tabs(["Upload Image", "Capture Photo"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        process_image(Image.open(uploaded_file))

with camera_tab:
    captured_photo = st.camera_input("Take a photo")

    if captured_photo is not None:
        process_image(Image.open(captured_photo))