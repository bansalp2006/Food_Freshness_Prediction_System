"""Train a Fresh/Rotten image classifier without TensorFlow (Python 3.14-friendly)."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "food_freshness.joblib"
CLASSES_PATH = MODEL_DIR / "class_names.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = (32, 32)

def label_from_path(path: Path):
    """Infer binary label from any ancestor folder named Fresh*/Rotten*."""
    for part in reversed(path.parts):
        name = part.lower().replace("_", "").replace("-", "").replace(" ", "")
        if "rotten" in name or "spoiled" in name:
            return "Rotten"
        if "fresh" in name:
            return "Fresh"
    return None

def features(image_path: Path):
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize(SIZE)
        pixels = np.asarray(im, dtype=np.float32) / 255.0
    # Downsampled RGB pixels plus per-channel histograms and simple color statistics.
    flat = pixels.reshape(-1)
    hist = []
    for channel in range(3):
        h, _ = np.histogram(pixels[:, :, channel], bins=16, range=(0, 1), density=False)
        hist.extend((h / max(1, h.sum())).tolist())
    stats = []
    for channel in range(3):
        plane = pixels[:, :, channel]
        stats.extend([float(plane.mean()), float(plane.std())])
    return np.concatenate([flat, np.asarray(hist, dtype=np.float32), np.asarray(stats, dtype=np.float32)])

def main():
    if not DATA_DIR.exists():
        raise SystemExit("Create data/ and put labeled image folders inside it.")
    files, labels = [], []
    skipped_unlabeled = 0
    for p in DATA_DIR.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_from_path(p.parent)
        if label is None:
            skipped_unlabeled += 1
            continue
        files.append(p)
        labels.append(label)
    counts = {c: labels.count(c) for c in ("Fresh", "Rotten")}
    if min(counts.values()) < 2:
        raise SystemExit(
            "Could not find at least 2 labeled images for BOTH Fresh and Rotten.\n"
            "Supported examples: data/Fresh/img.jpg, data/Rotten/img.jpg, or\n"
            "data/train/FreshApple/img.jpg and data/train/RottenApple/img.jpg.\n"
            "Folders for products without Fresh/Rotten/ Spoiled in their names are not labeled."
        )
    print(f"Found images: Fresh={counts['Fresh']}, Rotten={counts['Rotten']}")
    if skipped_unlabeled:
        print(f"Skipping {skipped_unlabeled} images whose folder names do not indicate Fresh/Rotten.")
    X_list, y_list = [], []
    for i, (p, label) in enumerate(zip(files, labels), start=1):
        try:
            X_list.append(features(p))
            y_list.append(label)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            print(f"Skipping unreadable image {p}: {exc}")
        if i % 500 == 0:
            print(f"Processed {i}/{len(files)} images...")
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list)
    if len(set(y)) < 2 or min(np.sum(y == c) for c in ("Fresh", "Rotten")) < 2:
        raise SystemExit("Not enough readable images in both classes. Check the dataset.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = ExtraTreesClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1,
                                 random_state=42, min_samples_leaf=1)
    print("Training ExtraTrees image classifier...")
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print("\nValidation accuracy: {:.2f}%".format(accuracy_score(y_val, pred) * 100))
    print(classification_report(y_val, pred, labels=["Fresh", "Rotten"], zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted; Fresh, Rotten):")
    print(confusion_matrix(y_val, pred, labels=["Fresh", "Rotten"]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    CLASSES_PATH.write_text(json.dumps(["Fresh", "Rotten"], indent=2), encoding="utf-8")
    print(f"\nSaved model: {MODEL_PATH}")
    print("Note: this is a baseline visual classifier, not a food-safety test.")

if __name__ == "__main__":
    main()
