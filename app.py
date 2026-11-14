
import os
import numpy as np
from PIL import Image
import streamlit as st

BACKEND = None
MODEL = None
TFLITE_INTERPRETER = None

IMG_SIZE = (224, 224)
MODEL_FILE = "plant_disease_no_finetune.h5"

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Classification")
st.write(
    "Upload a plant leaf image to predict its disease "
    "using a MobileNetV2-based deep learning model."
)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Model & Labels")

    use_tflite = st.toggle(
        "Use TFLite if available",
        value=False
    )

    model_path = st.text_input(
        "Model filename",
        value=MODEL_FILE
    )

    st.divider()

    st.subheader("Class Labels")

    labels_source = st.radio(
        "Provide labels via:",
        ["Paste list", "labels.txt file"],
        index=1
    )

    pasted_labels = ""
    label_file = None

    if labels_source == "Paste list":
        pasted_labels = st.text_area(
            "Paste comma-separated labels",
            placeholder="Enter labels in training order"
        )
    else:
        label_file = st.file_uploader(
            "Upload labels_15.txt",
            type=["txt"]
        )

    show_gradcam = st.checkbox(
        "Show Grad-CAM (Keras only)",
        value=False
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        0.0, 1.0, 0.3, 0.01
    )


# ---------------- LABEL UTILITIES ----------------

def load_labels():
    if labels_source == "Paste list":
        return [
            x.strip()
            for x in pasted_labels.split(",")
            if x.strip()
        ]

    if labels_source == "labels.txt file" and label_file:
        content = label_file.getvalue().decode("utf-8")
        return [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

    return []


# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_image(pil_img, size=IMG_SIZE):
    img = pil_img.convert("RGB").resize(size)

    x = np.array(img).astype("float32")
    x = x / 255.0

    x = np.expand_dims(x, axis=0)

    return x


# ---------------- MODEL LOADING ----------------

@st.cache_resource(show_spinner=True)
def load_keras_model(path):
    from tensorflow import keras

    model = keras.models.load_model(
        path,
        compile=False
    )

    return model


@st.cache_resource(show_spinner=True)
def load_tflite_model(path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=path
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def ensure_model():
    global BACKEND, MODEL, TFLITE_INTERPRETER

    if not os.path.exists(model_path):
        st.error(
            f"Model file not found: {model_path}. "
            "Check that the model is in the same folder as app.py."
        )
        return

    if use_tflite and model_path.endswith(".tflite"):
        try:
            interpreter, in_det, out_det = load_tflite_model(
                model_path
            )

            BACKEND = "tflite"
            TFLITE_INTERPRETER = (interpreter, in_det, out_det)
            MODEL = None
            return

        except Exception as e:
            st.error(f"TFLite loading failed: {e}")
            return

    try:
        MODEL = load_keras_model(model_path)
        BACKEND = "keras"
        TFLITE_INTERPRETER = None

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        BACKEND = None


# ---------------- PREDICTION ----------------

def predict_with_keras(model, x):
    predictions = model.predict(x, verbose=0)
    return predictions[0]


def predict_with_tflite(interpreter, input_details, output_details, x):
    interpreter.set_tensor(
        input_details[0]["index"],
        x.astype(input_details[0]["dtype"])
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    return output[0]


def get_topk(probs, labels, k=5):
    indices = np.argsort(probs)[::-1][:k]

    results = []

    for i in indices:
        name = labels[i] if i < len(labels) else f"class_{i}"

        results.append(
            (int(i), name, float(probs[i]))
        )

    return results


# ---------------- MAIN APP ----------------

ensure_model()

uploaded = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded is not None:

    try:
        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

        labels = load_labels()

        if BACKEND is None:
            st.stop()

        if not labels:
            st.warning(
                "Please upload labels_15.txt to display disease names."
            )
            st.stop()

        st.caption(f"Using backend: {BACKEND.upper()}")

        x = preprocess_image(image)

        with st.spinner("Analyzing leaf image..."):

            if BACKEND == "keras":
                probs = predict_with_keras(MODEL, x)

            else:
                interpreter, in_det, out_det = TFLITE_INTERPRETER

                probs = predict_with_tflite(
                    interpreter,
                    in_det,
                    out_det,
                    x
                )

        # Validate labels AFTER prediction
        if len(labels) != len(probs):
            st.error(
                f"Label mismatch: Model predicts {len(probs)} classes, "
                f"but {len(labels)} labels were provided. "
                "Please upload the correct labels_15.txt file."
            )
            st.stop()

        topk = get_topk(
            probs,
            labels,
            k=min(5, len(probs))
        )

        best_idx, best_name, best_conf = topk[0]

        st.subheader("Prediction")

        st.metric(
            label="Predicted Disease",
            value=best_name,
            delta=f"{best_conf * 100:.2f}% confidence"
        )

        st.write("### Top 5 Predictions")

        st.table({
            "Rank": list(range(1, len(topk) + 1)),
            "Class": [r[1] for r in topk],
            "Confidence": [
                f"{r[2] * 100:.2f}%"
                for r in topk
            ]
        })

        if best_conf < conf_threshold:
            st.warning(
                "Low confidence prediction. Try uploading a clearer leaf image."
            )

        # ---------------- GRAD-CAM ----------------

        if BACKEND == "keras" and show_gradcam:
            st.info(
                "Grad-CAM is optional and is not enabled in this initial version."
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")


else:
    st.info("Upload a leaf image to get started.")

st.divider()

st.caption(
    "Plant Disease Classification | MobileNetV2 Transfer Learning"
)
