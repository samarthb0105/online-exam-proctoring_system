# dataset_utils.py
import os
import cv2
import numpy as np
from tqdm import tqdm

def build_image_list(root_dir, classes=("open","closed")):
    items = []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg",".png",".jpeg")):
                items.append((os.path.join(cls_dir, fname), cls))
    return items

def preprocess_eye(img_path, size=(64,64)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: 
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, -1)
    return img

def load_dataset(root_dir, classes=("open","closed")):
    items = build_image_list(root_dir, classes)
    X, y = [], []
    for p, cls in tqdm(items):
        img = preprocess_eye(p)
        if img is None: continue
        X.append(img)
        y.append(0 if cls == "open" else 1)
    X = np.array(X)
    y = np.array(y)
    return X, y
