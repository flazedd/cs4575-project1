import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import implementations.constants as constants

print("Using jax", jax.__version__)

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


def set_seed(seed=42):
    return jax.random.PRNGKey(seed)


class TrainerModule:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.opt_state = None
        self.params = None
        self.main_key = set_seed(42)

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0,), (1,))  # Normalize
        ])

        self.train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)

        # Initialize model and optimizer
        self.model, self.params = self.initialize_model(self.main_key)
        self.optimizer = optax.adamw(1e-3)
        self.opt_state = self.optimizer.init(self.params)

        # JIT-compiled functions
        self.train_step_ = jax.jit(self.train_step)
        self.eval_step_ = jax.jit(self.eval)

        print("training JAX...")
        self.train(self.train_loader)
        self.eval()


    # Initialize model and parameters
    def initialize_model(self, rng):
        model = JaxCNN(hidden_channels=[5, 6], out_features=10)
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
    def train_epoch(self):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs = inputs.permute(0, 2, 3, 1)  # Rearrange to (B, H, W, C), since JAX uses Channel as last dim

            # Convert inputs and labels to JAX arrays
            inputs = jnp.array(inputs.numpy())
            labels = jnp.array(labels.numpy())

            # Perform a training step
            self.params, self.opt_state = self.train_step_(self.params, self.opt_state, inputs, labels)

            # Compute metrics
            logits = self.model.apply(self.params, inputs)
            loss = self.compute_loss(self.params, inputs, labels)
            running_loss += loss.item()
            predicted = jnp.argmax(logits, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = 100 * correct / total
        return train_loss, train_accuracy

    # Main training function
    def train(self, data_loader, epochs=constants.EPOCHS, learning_rate=1e-3):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Evaluation function
    def eval(self):
        correct = 0
        total = 0
        for inputs, labels in self.test_loader:
            inputs = inputs.permute(0, 2, 3, 1)  # Rearrange to (B, H, W, C), since JAX uses Channel as last dim

            # Convert inputs and labels to JAX arrays
            inputs = jnp.array(inputs.numpy())
            labels = jnp.array(labels.numpy())

            # Compute metrics
            logits = self.model.apply(self.params, inputs)
            predicted = jnp.argmax(logits, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")


def jax_task():
    TrainerModule()


if __name__ == "__main__":
    jax_task()