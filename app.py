
import streamlit as st
import numpy as np
import joblib
import cv2

from PIL import Image
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="IntelliFace AI", page_icon="🔍")
st.title("IntelliFace AI")
st.write("Facial recognition using InsightFace and SVM.")

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
    threshold = float(bundle.get("confidence_threshold", 60.0))
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()


def predict_face(face):
    embedding = face.embedding.reshape(1, -1)

    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(embedding)[0]
        best_idx = int(np.argmax(probabilities))
        score = float(probabilities[best_idx]) * 100
        encoded_label = clf.classes_[best_idx]
    else:
        encoded_label = clf.predict(embedding)[0]
        score = 100.0

    name = str(le.inverse_transform([encoded_label])[0])

    if score < threshold:
        return "Unknown", score

    return name, score


mode = st.radio(
    "Choose how to recognize faces",
    ["Upload an image", "Live webcam detection"],
    horizontal=True
)

if mode == "Upload an image":
    uploaded = st.file_uploader(
        "Upload a face photo",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded photo", use_container_width=True)

        # InsightFace expects BGR
        img = np.array(image)[:, :, ::-1].copy()
        faces = face_app.get(img)

        if not faces:
            st.warning("No face detected.")
        else:
            for i, face in enumerate(faces):
                name, score = predict_face(face)
                st.write(
                    f"Face {i + 1}: **{name}** — "
                    f"classifier score: {score:.1f}%"
                )

else:
    st.info(
        "Allow camera access when your browser asks. "
        "Keep your face visible to the camera."
    )

    class FaceRecognitionProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            faces = face_app.get(img)

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                name, score = predict_face(face)

                color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(
                    img, (x1, y1), (x2, y2), color, 2
                )
                label = f"{name} ({score:.0f}%)"

                cv2.putText(
                    img, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    import av

    webrtc_streamer(
        key="intelliface-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceRecognitionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        async_processing=True
    )
