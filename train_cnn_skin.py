import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model for skin segmentation
def build_skin_segmentation_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training the CNN model
def train_skin_segmentation_model():
    # Load preprocessed data
    images = np.load('./datasets/skin/samples_images.npy')
    labels = np.load('./datasets/skin/samples_labels.npy')

    # Ensure the label array has the correct shape
    labels = labels.reshape(labels.shape[0], -1).mean(axis=1)  # Binary classification: skin (1) or non-skin (0)
    labels = (labels > 0.5).astype(int)  # Convert to binary labels (0 or 1)

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    print("Train data shape:", X_train.shape, y_train.shape)
    print("Validation data shape:", X_val.shape, y_val.shape)

    # Build the model
    model = build_skin_segmentation_model()

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Save the trained model
    model.save('./models/skin_segmentation_model.h5')
    print("Model saved as 'skin_segmentation_model.h5'")

if __name__ == "__main__":
    train_skin_segmentation_model()
