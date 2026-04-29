import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ======================================
#  FAST DATASET PATH
# ======================================
DATASET_PATH = r"E:\online_Exam_proctoring_system\Datasets\MRLEye_data"

# ======================================
#  FAST CONFUSION MATRIX SAVER
# ======================================
def save_cm(cm, title, filename):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()


# ======================================
#  ULTRA–FAST DATA LOADER
#  Loads 2500 images from each class (≈ 5k total)
#  Resizes to 32x32 for speed
# ======================================
def load_dataset():
    images, labels = [], []

    for folder in ["open", "close"]:
        label = 1 if folder == "open" else 0
        folder_path = os.path.join(DATASET_PATH, folder)

        img_list = os.listdir(folder_path)[:2500]  # FAST LIMIT

        for img_name in tqdm(img_list, desc=f"Loading {folder}"):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (32, 32))  # FAST SIZE
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# ======================================
#  LOAD DATA ULTRA FAST
# ======================================
print("🚀 Loading dataset...")
X, y = load_dataset()
print(f"Dataset Loaded: {len(X)} images\n")


# FLATTEN FOR ML MODELS
X_flat = X.reshape(len(X), -1)

# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42
)


# ======================================
# 1️⃣ SUPPORT VECTOR MACHINE (FAST)
# ======================================
print("⚡ Training SVM...")
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
save_cm(confusion_matrix(y_test, svm_pred), "SVM Confusion Matrix", "SVM_fast_cm.png")

print(f"SVM Accuracy: {svm_acc:.4f}")


# ======================================
# 2️⃣ LOGISTIC REGRESSION (VERY FAST)
# ======================================
print("\n⚡ Training Logistic Regression...")
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)

log_pred = logreg.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
save_cm(confusion_matrix(y_test, log_pred), "LogReg Confusion Matrix", "LOGREG_fast_cm.png")

print(f"Logistic Regression Accuracy: {log_acc:.4f}")


# ======================================
# 3️⃣ KNN (SUPER FAST)
# ======================================
print("\n⚡ Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
save_cm(confusion_matrix(y_test, knn_pred), "KNN Confusion Matrix", "KNN_fast_cm.png")

print(f"KNN Accuracy: {knn_acc:.4f}")


# ======================================
# 4️⃣ RANDOM FOREST (FAST)
# ======================================
print("\n⚡ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
save_cm(confusion_matrix(y_test, rf_pred), "RF_fast_cm.png", "RF_fast_cm.png")

print(f"Random Forest Accuracy: {rf_acc:.4f}")


# ======================================
# 5️⃣ CNN (OPTIMIZED ULTRA FAST)
# ======================================
print("\n⚡ Training CNN (1 Epoch)...")

# CNN needs original shape
X_cnn = X / 255.0
X_cnn = X_cnn.reshape(-1, 32, 32, 1)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42
)

cnn = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(X_train_cnn, y_train_cnn, epochs=1, batch_size=32, verbose=1)

cnn_pred = (cnn.predict(X_test_cnn) > 0.5).astype(int)
cnn_acc = accuracy_score(y_test_cnn, cnn_pred)
save_cm(confusion_matrix(y_test_cnn, cnn_pred), "CNN Fast Confusion Matrix", "CNN_fast_cm.png")

print(f"CNN Accuracy: {cnn_acc:.4f}")

print("\n================================================")
print("    🚀 ALL MODELS TRAINED ULTRA FAST SUCCESSFULLY 🚀")
print("================================================")
