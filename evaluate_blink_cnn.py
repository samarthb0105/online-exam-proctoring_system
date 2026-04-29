import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CHANGE THIS TO YOUR DATASET
# ---------------------------
DATASET_PATH = r"E:\online_Exam_proctoring_system\Datasets\MRLEye_data"
CLASSES = ["close", "open"]

# ---------------------------
# LOAD DATASET FUNCTION
# ---------------------------
def load_dataset(path, target_size):
    X = []
    y = []

    for label_idx, label in enumerate(CLASSES):
        folder = os.path.join(path, label)
        if not os.path.exists(folder):
            print(f"[WARNING] Folder missing → {folder}")
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert to grayscale if model expects 1 channel
            if target_size[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, (target_size[1], target_size[0]))
            img = img / 255.0
            X.append(img)
            y.append(label_idx)

    X = np.array(X).reshape(-1, target_size[0], target_size[1], target_size[2])
    y = np.array(y)

    print(f"[INFO] Loaded {len(X)} images.")
    return X, y

# ---------------------------
# SAVE CONFUSION MATRIX
# ---------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix – Blink CNN")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")

# ---------------------------
# MAIN EVALUATION
# ---------------------------
if __name__ == "__main__":
    # Load trained model
    print("[INFO] Loading trained model...")
    model = load_model("results/models/blink_cnn.h5")
    input_shape = model.input_shape[1:]  # exclude batch dimension
    print(f"[INFO] Model expects input shape: {input_shape}")

    # Load dataset
    print("[INFO] Loading dataset...")
    X, y = load_dataset(DATASET_PATH, target_size=input_shape)

    # Make predictions
    print("[INFO] Making predictions...")
    preds = model.predict(X)
    preds = (preds > 0.5).astype("int32").flatten()

    # Classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(y, preds, target_names=CLASSES))

    # Save confusion matrix
    plot_confusion_matrix(y, preds, "results/blink_confusion_matrix.png")

    print("\n[FINAL] Evaluation complete!")
