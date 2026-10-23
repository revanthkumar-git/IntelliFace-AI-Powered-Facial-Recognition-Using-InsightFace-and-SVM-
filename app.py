import streamlit as st
import numpy as np
import joblib
from PIL import Image
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Face Recognition", page_icon="🔍")
st.title("Face Recognition App")
st.write("Upload a photo to identify a person from your trained classes.")

@st.cache_resource
def load_models():
    bundle = joblib.load("face_recognition_model.pkl")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    return bundle, face_app

try:
    bundle, face_app = load_models()
    clf = bundle["classifier"]
    le = bundle["label_encoder"]
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # InsightFace expects BGR image data
    img = np.array(image)[:, :, ::-1].copy()
    faces = face_app.get(img)

    if not faces:
        st.warning("No face detected. Try another photo.")
    else:
        for i, face in enumerate(faces):
            embedding = face.embedding.reshape(1, -1)

            if hasattr(clf, "predict_proba"):
                probabilities = clf.predict_proba(embedding)[0]
                best_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[best_idx]) * 100
                prediction = clf.classes_[best_idx]

                st.subheader(f"Face {i + 1}")
                if confidence >= bundle.get("confidence_threshold", 60.0):
                    st.success(
                        f"Predicted class: {le.inverse_transform([prediction])[0]}"
                    )
                    st.write(f"Classifier probability: {confidence:.1f}%")
                else:
                    st.warning("Prediction uncertain.")
            else:
                prediction = clf.predict(embedding)[0]
                name = le.inverse_transform([prediction])[0]
                st.write(f"Face {i + 1}: {name}")
