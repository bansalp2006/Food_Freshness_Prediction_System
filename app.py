from pathlib import Path
import json
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "food_freshness.joblib"
CLASSES_PATH = ROOT / "models" / "class_names.json"
SIZE = (32, 32)

def image_features(image):
    image = ImageOps.exif_transpose(image).convert("RGB").resize(SIZE)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    flat = pixels.reshape(-1)
    hist = []
    for channel in range(3):
        h, _ = np.histogram(pixels[:, :, channel], bins=16, range=(0, 1), density=False)
        hist.extend((h / max(1, h.sum())).tolist())
    stats = []
    for channel in range(3):
        plane = pixels[:, :, channel]
        stats.extend([float(plane.mean()), float(plane.std())])
    return np.concatenate([flat, np.asarray(hist, dtype=np.float32), np.asarray(stats, dtype=np.float32)]).reshape(1, -1)

st.set_page_config(page_title="Food Freshness AI", page_icon="🍎", layout="centered")
st.title("🍎 Food Freshness Prediction System")
st.write("Upload a fruit or vegetable image for an ML-based Fresh/Rotten prediction.")
st.info("Educational demo only. Appearance alone cannot reliably determine whether food is safe to eat.")
if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Trained model not found. Add labeled images and run `python train.py` first.")
    st.code("data/Fresh/     # images labeled Fresh\ndata/Rotten/    # images labeled Rotten\n\npython train.py\nstreamlit run app.py", language="text")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
try:
    model, class_names = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded:
    try:
        image = Image.open(uploaded)
        image = ImageOps.exif_transpose(image).convert("RGB")
        try:
            st.image(image, caption="Uploaded image", width="stretch")
        except TypeError:
            st.image(image, caption="Uploaded image", use_container_width=True)
        x = image_features(image)
        probabilities = model.predict_proba(x)[0]
        class_index = int(np.argmax(probabilities))
        predicted = str(model.classes_[class_index])
        confidence = float(probabilities[class_index])
        st.subheader("Prediction")
        if predicted == "Fresh":
            st.success(f"Predicted class: {predicted}")
        else:
            st.warning(f"Predicted class: {predicted}")
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        scores = {str(c): float(p) for c, p in zip(model.classes_, probabilities)}
        st.caption("Fresh score: {:.1f}% · Rotten score: {:.1f}%".format(scores.get("Fresh", 0)*100, scores.get("Rotten", 0)*100))
        st.warning("Confidence is not a guarantee. Do not use this result alone to decide whether food is safe to eat.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
with st.expander("How this project works"):
    st.markdown("""
    1. The training script reads images from folders labeled Fresh or Rotten (including category folders such as FreshApple and RottenBanana).
    2. It extracts simple image/color features and trains a scikit-learn ExtraTrees classifier.
    3. A held-out validation split reports accuracy and a classification report.
    4. Upload a new image here after training to see a prediction.

    This lightweight model is not a deep-learning model and may be less accurate than a well-trained transfer-learning model. Keep near-duplicate images out of both training and validation sets for more realistic evaluation.
    """)
