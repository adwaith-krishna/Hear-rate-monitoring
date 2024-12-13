import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_skin_nonskin(file_path):
    # Load the dataset
    data = np.loadtxt(file_path)

    # Split into RGB values and labels
    X = data[:, :3]  # First 3 columns are RGB values
    y = data[:, 3]  # Fourth column is the label (1 for skin, 0 for non-skin)

    # Normalize the RGB values
    X = X / 255.0  # Scale to [0, 1]

    # Reshape into small image patches if needed
    # Here, creating dummy image patches of size (128x128x3) for demonstration
    # For example purposes, assume each 128x128 patch is a "batch" of rows
    num_pixels = X.shape[0]
    image_size = 128 * 128
    num_images = num_pixels // image_size

    X_images = X[:num_images * image_size].reshape(num_images, 128, 128, 3)
    y_images = y[:num_images * image_size].reshape(num_images, 128, 128, 1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_images, y_images, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Example usage
file_path = './datasets/skin/skin_nonskin.txt'
X_train, X_test, y_train, y_test = preprocess_skin_nonskin(file_path)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")
