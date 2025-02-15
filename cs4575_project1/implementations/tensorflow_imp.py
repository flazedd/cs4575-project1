import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a Sequential model
model = models.Sequential([
    # Flatten the 28x28 images into a 1D vector of 784 pixels
    layers.Flatten(input_shape=(28, 28)),

    # Add a Dense layer with 128 neurons and ReLU activation
    layers.Dense(128, activation='relu'),

    # Add a Dropout layer for regularization (50% drop rate)
    layers.Dropout(0.2),

    # Output layer with 10 neurons (one for each digit) and softmax activation
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Optionally: display a sample image and its predicted label
plt.imshow(x_test[0], cmap=plt.cm.binary)  # Display first test image
plt.show()

# Predict the label for the first image in the test set
predicted_label = np.argmax(model.predict(x_test[0:1]), axis=1)
print(f'Predicted label: {predicted_label[0]}')
