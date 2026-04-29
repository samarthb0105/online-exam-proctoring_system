import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_PATH = r"E:\online_Exam_proctoring_system\Datasets\MRLEye_data"

def load_dataset():
    images, labels = [], []

    for folder in ["open", "close"]:
        label = 1 if folder == "open" else 0
        folder_path = os.path.join(DATASET_PATH, folder)

        for img_name in tqdm(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(label)

    return np.array(images), np.array(labels)

# Load data
X, y = load_dataset()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("Training SVM...")
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("SVM Confusion Matrix")
plt.savefig("SVM_confusion_matrix.png")
plt.close()
