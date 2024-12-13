import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def build_skin_segmentation_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(1, (1, 1), activation='sigmoid')  # Pixel-wise binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_images = np.load('./datasets/skin/train_images.npy')
    train_labels = np.load('./datasets/skin/train_labels.npy') / 255.0  # Normalize
    test_images = np.load('./datasets/skin/test_images.npy')
    test_labels = np.load('./datasets/skin/test_labels.npy') / 255.0  # Normalize

    # Reshape labels for segmentation
    train_labels = train_labels.reshape(-1, 128, 128, 1)
    test_labels = test_labels.reshape(-1, 128, 128, 1)

    model = build_skin_segmentation_model()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=32)

    model.save('./models/skin_segmentation_model.h5')
    print("Model saved to './models/skin_segmentation_model.h5'")

if __name__ == "__main__":
    train_model()
