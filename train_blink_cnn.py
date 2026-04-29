# train_blink_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from dataset_utils import load_dataset
import argparse
import os

def build_model(input_shape=(64,64,1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32,3,activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(dataset_dir, epochs=10, batch_size=32, out_dir="results/models"):
    X, y = load_dataset(dataset_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(input_shape=X.shape[1:])
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "blink_cnn.h5"), save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    print("Training finished. Model saved to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to labeled dataset root (open/, closed/)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    main(args.dataset, epochs=args.epochs, batch_size=args.batch)
