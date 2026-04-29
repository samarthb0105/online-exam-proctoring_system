import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# SETTINGS
# ---------------------------
DATASET_PATH = r"E:\online_Exam_proctoring_system\Datasets\MRLEye_data"
CLASSES = ["close", "open"]       # Folder names must match exactly
IMG_SIZE = 64                     # Resize images to 64x64
SUBSET_SIZE_PER_CLASS = 500       # Load 500 images per class for fast demo
RESULTS_DIR = "results"           # Folder to save confusion matrix

# Create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# LOAD BALANCED SUBSET
# ---------------------------
def load_dataset(path, img_size, subset_size_per_class=500):
    X = []
    y = []

    for label_idx, label in enumerate(CLASSES):
        folder = os.path.join(path, label)
        if not os.path.exists(folder):
            print(f"[WARNING] Folder missing → {folder}")
            continue

        files = os.listdir(folder)
        if subset_size_per_class:
            files = files[:subset_size_per_class]  # Take subset per class

        for img_name in files:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            X.append(img)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] Loaded {len(X)} images (balanced subset).")
    return X, y

# ---------------------------
# PLOT & SAVE CONFUSION MATRIX
# ---------------------------
def plot_confusion(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Confusion matrix saved → {save_path}")
    plt.show()

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Load dataset subset
    X, y = load_dataset(DATASET_PATH, IMG_SIZE, subset_size_per_class=SUBSET_SIZE_PER_CLASS)

    # Flatten images for Random Forest
    X_flat = X.reshape(len(X), -1)

    # Initialize and train Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_flat, y)

    # Make predictions
    preds = rf.predict(X_flat)

    # Classification report
    print("\n[INFO] Random Forest Classification Report:")
    print(classification_report(y, preds, target_names=CLASSES))

    # Confusion matrix (display + save)
    cm_path = os.path.join(RESULTS_DIR, "rf_confusion_matrix.png")
    plot_confusion(y, preds, "Random Forest", save_path=cm_path)

    print("\n[FINAL] Random Forest evaluation complete!")
