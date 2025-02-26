import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
import cs4575_project1.implementations.constants as constants
import numpy as np
import tensorflow_datasets as tfds


print("Using jax", jax.__version__)

def set_seed(seed=42):
    return jax.random.PRNGKey(seed)

class JaxCNN(nn.Module):
    hidden_channels: list  # Hidden channels for two conv layers
    out_features: int  # Output dimension for final dense layer

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.hidden_channels[0], kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        x = nn.Conv(features=self.hidden_channels[1], kernel_size=(5, 5), padding=2)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten spatial dimensions
        x_out = nn.Dense(self.out_features)(x)

        return x_out
    
class TrainerModule:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.opt_state = None
        self.params = None
        self.main_key = set_seed(42)

        # Load MNIST dataset using TFDS
        mnist_data = tfds.load('mnist', as_supervised=True, data_dir='./data/jax')
        train_dataset = mnist_data['train']
        test_dataset = mnist_data['test']

        self.x_train, self.y_train = self.convert_tfds_to_jax(train_dataset)
        self.x_test, self.y_test = self.convert_tfds_to_jax(test_dataset)

        # Normalize images to [0, 1]
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # Initialize model and optimizer
        self.model, self.params = self.initialize_model(self.main_key)
        self.optimizer = optax.adamw(constants.LEARNING_RATE)
        self.opt_state = self.optimizer.init(self.params)

        # JIT-compiled functions
        self.train_step_ = jax.jit(self.train_step)
        self.eval_step_ = jax.jit(self.eval)

        print("training JAX...")
        self.train()
        self.eval()

    # Converts numpy from tensorflow_datasets to jnp
    def convert_tfds_to_jax(self, dataset):
            images, labels = [], []
            for img, lbl in tfds.as_numpy(dataset):
                images.append(img)
                labels.append(lbl)
            images = np.stack(images)  # Shape: (N, 28, 28, 1)
            labels = np.array(labels)  # Shape: (N,)
            return jnp.array(images), jnp.array(labels)

    # Initialize model and parameters
    def initialize_model(self, rng):
        model = JaxCNN(hidden_channels=[64, 128], out_features=10)
        params = model.init(rng, jnp.ones((1, 28, 28, 1)))
        return model, params

    # Define the loss function
    def compute_loss(self, params, inputs, labels):
        logits = self.model.apply(params, inputs)
        one_hot_labels = jax.nn.one_hot(labels, num_classes=10)  # Convert to one-hot, shape: (Batch, 10)
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss

    # Train one batch
    def train_step(self, params, opt_state, inputs, labels):
        def loss_fn(params):
            return self.compute_loss(params, inputs, labels)

        # Compute gradients
        grads = jax.grad(loss_fn)(params)

        # Update parameters and optimizer state
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    # Train one epoch
    def train_epoch(self, batch_size=constants.BATCH_SIZE):
        running_loss = 0.0
        correct = 0
        total = 0
        num_samples = self.x_train.shape[0]
        num_batches = num_samples // batch_size

        # Shuffle training data
        key, subkey = jax.random.split(self.main_key)
        indices = jax.random.permutation(subkey, num_samples)

        for i in range(num_batches):
            batch_idx = indices[i * batch_size:(i + 1) * batch_size]
            inputs = self.x_train[batch_idx]
            labels = self.y_train[batch_idx]

            # Perform training step
            self.params, self.opt_state = self.train_step_(self.params, self.opt_state, inputs, labels)

            # Compute metrics
            logits = self.model.apply(self.params, inputs)
            loss = self.compute_loss(self.params, inputs, labels)
            running_loss += loss.item()
            predicted = jnp.argmax(logits, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / num_batches
        train_accuracy = 100 * correct / total
        return train_loss, train_accuracy

    # Main training function
    def train(self, epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(batch_size)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Evaluation function
    def eval(self, batch_size=constants.BATCH_SIZE):
        correct = 0
        total = 0
        num_samples = self.x_test.shape[0]
        num_batches = num_samples // batch_size

        for i in range(num_batches):
            inputs = self.x_test[i * batch_size:(i + 1) * batch_size]
            labels = self.y_test[i * batch_size:(i + 1) * batch_size]

            logits = self.model.apply(self.params, inputs)
            predicted = jnp.argmax(logits, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")

def jax_jit_task():
    TrainerModule()


if __name__ == "__main__":
    jax_jit_task()