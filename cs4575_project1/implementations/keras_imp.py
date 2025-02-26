import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import cs4575_project1.implementations.constants as constants

def set_tensorflow_seed(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def keras_task():
    # Set seed
    set_tensorflow_seed(42)

    # 1️⃣ Load & Preprocess MNIST Dataset
    # Download MNIST if not already downloaded
    dataset_path = tf.keras.utils.get_file(
    fname="mnist.npz",  # Mnist
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    cache_dir="./data/keras"  # Download directory
    )
    #  Load the data using numpy
    data = np.load(dataset_path)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Normalize pixel values to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Add channel dimension for TensorFlow (for Conv2D)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

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
    layers.Dense(10, activation="softmax")  # 10 output classes
])

    # 3️⃣ Compile the Model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # 4️⃣ Train the Model
    # Train the model on CPU
    device = '/CPU:0'
    with tf.device(device):
        model.fit(x_train, y_train, epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE,
                  validation_data=(x_test, y_test))

    # 5️⃣ Evaluate on Test Data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    keras_task()