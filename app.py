import io
import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Face Recognition System", page_icon="👤", layout="wide")
st.title("👤 Face Recognition System")
st.write("Recognize faces from an uploaded image or a photo captured with your camera.")

CONFIDENCE_THRESHOLD = 0.40
MIN_PROBABILITY_GAP = 0.15


@st.cache_resource
def load_models():
    clf = joblib.load("face_svm.pkl")
    le = joblib.load("label_encoder.pkl")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return clf, le, face_app


try:
    with st.spinner("Loading face recognition model..."):
        clf, le, face_app = load_models()
except FileNotFoundError as exc:
    st.error(
        f"Model file not found: `{exc.filename}`. Put `face_svm.pkl` and "
        "`label_encoder.pkl` in the same folder as `app.py`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the face recognition model: {exc}")
    st.stop()


def recognize_faces(image, threshold=CONFIDENCE_THRESHOLD, min_gap=MIN_PROBABILITY_GAP):
    """Return an annotated BGR image and the number of detected faces."""
    faces = face_app.get(image)
    annotated = image.copy()

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        embedding = face.normed_embedding.reshape(1, -1)
        probabilities = clf.predict_proba(embedding)[0]
        order = np.argsort(probabilities)[::-1]

        best_idx = int(order[0])
        best_prob = float(probabilities[best_idx])
        second_prob = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        gap = best_prob - second_prob

        if best_prob >= threshold and gap >= min_gap:
            prediction = clf.classes_[best_idx]
            if isinstance(prediction, (int, np.integer)):
                name = str(le.inverse_transform([int(prediction)])[0])
            else:
                name = str(prediction)
            label = f"{name} ({best_prob:.0%})"
            color = (0, 180, 0)
        else:
            label = f"Unknown ({best_prob:.0%} top match)"
            color = (0, 0, 220)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
        )

    return annotated, len(faces)


def image_to_png_bytes(rgb_array):
    output = io.BytesIO()
    Image.fromarray(rgb_array).save(output, format="PNG")
    return output.getvalue()


def process_image(pil_image):
    try:
        rgb_image = np.array(pil_image.convert("RGB"))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting and recognizing faces..."):
            annotated, total_faces = recognize_faces(bgr_image)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        left, right = st.columns([3, 1], gap="large")

        with left:
            st.image(annotated_rgb, caption="Recognition result", use_container_width=True)

        with right:
            st.metric("Faces detected", total_faces)
            st.caption(
                "The percentage is the classifier's top probability, not a "
                "guaranteed measure of real-world accuracy."
            )
            st.download_button(
                "Download result",
                data=image_to_png_bytes(annotated_rgb),
                file_name="face_recognition_result.png",
                mime="image/png",
                use_container_width=True,
            )

        if total_faces == 0:
            st.warning(
                "No faces were detected. Try a clearer image with good lighting "
                "and faces looking toward the camera."
            )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")


st.divider()
upload_tab, camera_tab = st.tabs(["📁 Upload image", "📷 Capture photo"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG.",
    )
    if uploaded_file is not None:
        try:
            process_image(Image.open(uploaded_file))
        except Exception as exc:
            st.error(f"Unable to open this image: {exc}")

with camera_tab:
    captured_photo = st.camera_input("Take a photo")
    if captured_photo is not None:
        try:
            process_image(Image.open(captured_photo))
        except Exception as exc:
            st.error(f"Unable to open the captured photo: {exc}")

with st.expander("Recognition settings"):
    st.write(f"Confidence threshold: **{CONFIDENCE_THRESHOLD:.0%}**")
    st.write(f"Minimum gap between top two matches: **{MIN_PROBABILITY_GAP:.0%}**")
    st.caption(
        "These values are fixed in app.py. Change them only after testing "
        "known identities and people outside your dataset."
    )

st.caption(
    "Privacy note: process images responsibly and get permission before "
    "using other people's photos."
)

