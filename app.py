import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# ----------------------------
# Load model (cached so it only loads once)
# ----------------------------
MODEL_PATH = "1.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

MODEL = load_model()

CLASS_NAMES = [
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Septoria_leaf_spot',
    'Tomato__Target_Spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato__Tomato_mosaic_virus'
]

# ----------------------------
# Helper functions
# ----------------------------
def read_file_as_image(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image)
    return image

def predict(image: np.ndarray):
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

def format_label(label: str) -> str:
    return label.replace("Tomato_", "").replace("_", " ").replace("__", " ").strip()

# ----------------------------
# UI
# ----------------------------
st.title("🍅 Tomato Leaf Disease Detection")
st.write(
    "Upload a photo of a tomato leaf and the model will predict whether it's "
    "healthy or affected by a disease."
)

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            try:
                image = read_file_as_image(uploaded_file)
                predicted_class, confidence = predict(image)

                if confidence < 0.75:
                    st.warning(
                        "This doesn't look like a tomato leaf, or the image is unclear. "
                        "Please upload a clear photo of a tomato leaf."
                    )
                    st.write(f"Confidence: {confidence:.2%}")
                else:
                    label = format_label(predicted_class)
                    if predicted_class == "Tomato_healthy":
                        st.success(f"✅ Prediction: **{label}**")
                    else:
                        st.error(f"⚠️ Prediction: **{label}**")
                    st.write(f"Confidence: {confidence:.2%}")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
else:
    st.info("Please upload an image to get started.")

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit · by Muntazir Fatema")
