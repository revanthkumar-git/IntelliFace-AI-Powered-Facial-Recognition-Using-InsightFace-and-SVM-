# Face Recognition System

A face recognition web application built with **Streamlit**,
**InsightFace**, and an **SVM classifier**. Upload an image or capture a
photo with your camera to detect faces and identify people from the
trained dataset.

## Features

-   Upload JPG, JPEG, or PNG images.
-   Capture a photo using your device camera.
-   Detect faces using InsightFace (`buffalo_l`).
-   Recognize detected faces using a trained SVM classifier.
-   Display the predicted name and classifier probability.
-   Mark uncertain matches as **Unknown** using confidence and
    probability-gap thresholds.
-   Download the annotated recognition result.

## Project structure

Keep these files together in the project directory:

``` text
face-recognition-system/
├── app.py
├── face_svm.pkl
├── label_encoder.pkl
├── requirements.txt
├── packages.txt
└── README.md
```

The two `.pkl` files are required by the current application.

## Requirements

-   Python
-   The Python packages listed in `requirements.txt`
-   The system packages listed in `packages.txt` (needed in some
    Linux/cloud deployments)

## Installation and running locally

1.  Clone the repository:

    ``` bash
    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd face-recognition-system
    ```

2.  (Recommended) Create and activate a virtual environment:

    ``` bash
    python -m venv .venv
    ```

    On Windows:

    ``` bash
    .venv\Scripts\activate
    ```

    On macOS/Linux:

    ``` bash
    source .venv/bin/activate
    ```

3.  Install the Python dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Make sure `app.py`, `face_svm.pkl`, and `label_encoder.pkl` are in
    the same directory.

5.  Start the application:

    ``` bash
    streamlit run app.py
    ```

6.  Open the local URL printed by Streamlit in your browser.

## Model files: important

-   **`face_svm.pkl`** contains the trained SVM classifier.
-   **`label_encoder.pkl`** maps the classifier's numeric class labels
    back to the corresponding person names.

The current `app.py` loads both files. The classifier's classes are
numeric, and the application uses the label encoder's
`inverse_transform()` to obtain names. **Do not remove
`label_encoder.pkl` unless you also change the model/application so that
the encoder is no longer needed.**

Keep both files in the same folder as `app.py`, and use the exact
filenames shown above.

## Recognition settings

The current application uses these fixed settings in `app.py`:

-   Confidence threshold: `0.40` (40%)
-   Minimum gap between the top two class probabilities: `0.15` (15%)

A face is assigned a person's name only when both conditions are met.
Otherwise, it is displayed as **Unknown**. These thresholds should be
evaluated with test images before changing them.

## Notes

-   The application uses InsightFace's `buffalo_l` model and the CPU
    execution provider.
-   The first run may require downloading the InsightFace model assets.
-   The displayed probability is the classifier's top probability; it is
    not a guarantee of real-world recognition accuracy.
-   The supplied model files were saved with scikit-learn 1.6.1. Using a
    different scikit-learn version may cause compatibility warnings or
    issues when loading them. Prefer the training environment's version,
    or test the models carefully after changing versions.
-   `packages.txt` contains Linux system dependencies;
    `requirements.txt` contains Python dependencies.

## Privacy

Use face images responsibly. Obtain permission before processing other
people's photos, and handle uploaded images and trained face data
securely.
