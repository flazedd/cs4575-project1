import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from torchvision import datasets, transforms
import cs4575_project1.implementations.constants as constants

def set_tensorflow_seed(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def keras_task():
    # Example usage:
    set_tensorflow_seed(42)
    # 1️⃣ Load & Preprocess MNIST Dataset
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #
    # # Normalize pixel values to [0, 1]
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    #
    # # Add channel dimension (for Conv2D)
    # x_train = x_train.reshape(-1, 28, 28, 1)
    # x_test = x_test.reshape(-1, 28, 28, 1)
    # Load MNIST dataset using PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0,), (1,))  # Normalize
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Initialize variables to store min and max pixel values
    # min_pixel_value = float('inf')  # Start with a large value for min
    # max_pixel_value = float('-inf')  # Start with a small value for max
    #
    # # Iterate through the entire training and test datasets
    # for i in range(len(train_dataset)):
    #     image, _ = train_dataset[i]  # Get the image, ignoring the label
    #     min_pixel_value = min(min_pixel_value, image.min())  # Update min pixel value
    #     max_pixel_value = max(max_pixel_value, image.max())  # Update max pixel value
    #
    # for i in range(len(test_dataset)):
    #     image, _ = test_dataset[i]  # Get the image, ignoring the label
    #     min_pixel_value = min(min_pixel_value, image.min())  # Update min pixel value
    #     max_pixel_value = max(max_pixel_value, image.max())  # Update max pixel value
    #
    # # Print the results
    # print(f"Smallest pixel value encountered: {min_pixel_value}")
    # print(f"Largest pixel value encountered: {max_pixel_value}")

    # sys.exit(0)

    # Convert PyTorch tensors to NumPy arrays
    x_train = np.array([np.array(train_dataset[i][0]) for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    x_test = np.array([np.array(test_dataset[i][0]) for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

    # 2️⃣ Define CNN Model
    model = keras.Sequential([
    # First Convolutional Layer
    layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2, strides=2),

    # Second Convolutional Layer
    layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2, strides=2),

    # Flatten Layer (equivalent to x.view(x.size(0), -1) in PyTorch)
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(10, activation="softmax")  # Assuming the output has 10 classes, like MNIST
])

    # 3️⃣ Compile the Model
    # model.compile(optimizer="adam",
    #               loss="sparse_categorical_crossentropy",
    #               metrics=["accuracy"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=constants.LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4️⃣ Train the Model
    # Train the model on CPU
    device = '/CPU:0'
    with tf.device(device):
        model.fit(x_train, y_train, epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE, validation_data=(x_test, y_test))

    # 5️⃣ Evaluate on Test Data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    keras_task()