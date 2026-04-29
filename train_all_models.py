import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ================================
#  Dataset Path
# ================================
DATASET_PATH = r"E:\online_Exam_proctoring_system\Datasets\MRLEye_data"

# ================================
#  Load Dataset
# ================================
print("Loading dataset...")

images = []
labels = []

for label_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label_folder)
    if not os.path.isdir(folder_path):
        continue
    
    label = 0 if "close" in label_folder.lower() else 1  # 0 = closed, 1 = open
    
    for img_name in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)
        except:
            pass

images = np.array(images)
labels = np.array(labels)

print("Dataset Loaded Successfully!")
print(f"Total Images: {len(images)}")

# ================================
#  Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

X_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# ================================
#  SAVE CONFUSION MATRIX
# ================================
def save_cm(cm, title, filename):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()

# ================================
# 1️⃣ CNN Model
# ================================
print("\nTraining CNN...")

X_cnn = images / 255.0
X_cnn = X_cnn.reshape(-1, 64, 64, 1)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, labels, test_size=0.2, random_state=42
)

cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(X_train_cnn, y_train_cnn, epochs=3, batch_size=32, verbose=1)

cnn_pred = (cnn.predict(X_test_cnn) > 0.5).astype(int)
cnn_cm = confusion_matrix(y_test_cnn, cnn_pred)
cnn_acc = accuracy_score(y_test_cnn, cnn_pred)

save_cm(cnn_cm, "CNN Confusion Matrix", "CNN_Blink_cm.png")
print(f"CNN Accuracy: {cnn_acc:.4f}")

# ================================
# 2️⃣ SVM Model
# ================================
print("\nTraining SVM...")

svm = SVC(kernel="linear")
svm.fit(X_flat, y_train)

svm_pred = svm.predict(X_test_flat)
svm_cm = confusion_matrix(y_test, svm_pred)
svm_acc = accuracy_score(y_test, svm_pred)

save_cm(svm_cm, "SVM Confusion Matrix", "SVM_Blink_cm.png")
print(f"SVM Accuracy: {svm_acc:.4f}")

# ================================
# 3️⃣ Random Forest
# ================================
print("\nTraining Random Forest...")

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_flat, y_train)

rf_pred = rf.predict(X_test_flat)
rf_cm = confusion_matrix(y_test, rf_pred)
rf_acc = accuracy_score(y_test, rf_pred)

save_cm(rf_cm, "Random Forest Confusion Matrix", "RF_Blink_cm.png")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# ================================
# 4️⃣ Logistic Regression
# ================================
print("\nTraining Logistic Regression...")

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_flat, y_train)

log_pred = log_reg.predict(X_test_flat)
log_cm = confusion_matrix(y_test, log_pred)
log_acc = accuracy_score(y_test, log_pred)

save_cm(log_cm, "Logistic Regression Confusion Matrix", "LOGREG_Blink_cm.png")
print(f"Logistic Regression Accuracy: {log_acc:.4f}")

# ================================
# 5️⃣ KNN Model
# ================================
print("\nTraining KNN...")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_flat, y_train)

knn_pred = knn.predict(X_test_flat)
knn_cm = confusion_matrix(y_test, knn_pred)
knn_acc = accuracy_score(y_test, knn_pred)

save_cm(knn_cm, "KNN Confusion Matrix", "KNN_Blink_cm.png")
print(f"KNN Accuracy: {knn_acc:.4f}")

print("\n===============================")
print(" ALL MODELS TRAINED SUCCESSFULLY ")
print("===============================")
