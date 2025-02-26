import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import cs4575_project1.implementations.constants as constants

print("Using jax", jax.__version__)

def set_seed(seed=42):
    return jax.random.PRNGKey(seed)

def jax_task():
    # Example usage:
    main_key = set_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0,), (1,))  # Normalize
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, shuffle=False)

    # Force CPU device
    cpu_device = jax.devices("cpu")[0]  # Get the CPU device

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

    # Initialize model and parameters
    def initialize_model(rng):
        model = JaxCNN(hidden_channels=[64, 128], out_features=10)
        params = model.init(rng, jnp.ones((1, 28, 28, 1)))
        return model, params

    # Define the loss function
    def compute_loss(model, params, inputs, labels):
        logits = model.apply(params, inputs)
        one_hot_labels = jax.nn.one_hot(labels, num_classes=10)  # Convert to one-hot, shape: (Batch, 10)
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss

    # Train one batch
    # @jax.jit
    def train_step(model, params, opt_state, inputs, labels, optimizer):
        def loss_fn(params):
            return compute_loss(model, params, inputs, labels)

        # Compute gradients
        grads = jax.grad(loss_fn)(params)

        # Update parameters and optimizer state
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    # Train one epoch
    def train_epoch(model, params, opt_state, train_loader, optimizer):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.permute(0, 2, 3, 1)  # Rearrange to (B, H, W, C), since JAX uses Channel as last dim

            # Convert inputs and labels to JAX arrays
            inputs = jnp.array(inputs.numpy())
            labels = jnp.array(labels.numpy())

            # Place inputs and labels explicitly on CPU
            inputs = jax.device_put(inputs, cpu_device)
            labels = jax.device_put(labels, cpu_device)

            # Perform a training step
            params, opt_state = train_step(model, params, opt_state, inputs, labels, optimizer)

            # Compute metrics
            logits = model.apply(params, inputs)
            loss = compute_loss(model, params, inputs, labels)
            running_loss += loss.item()
            predicted = jnp.argmax(logits, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        return params, opt_state, train_loss, train_accuracy

    # Main training function
    def train(train_loader, epochs=constants.EPOCHS, learning_rate=constants.LEARNING_RATE):
        model, params = initialize_model(main_key)
        optimizer = optax.adamw(learning_rate)
        opt_state = optimizer.init(params)

        for epoch in range(epochs):
            params, opt_state, train_loss, train_accuracy = train_epoch(
                model, params, opt_state, train_loader, optimizer
            )
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        return model, params, optimizer, opt_state

    model, params, optimizer, opt_state = train(train_loader)

    # Eval step
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.permute(0, 2, 3, 1)  # Rearrange to (B, H, W, C), since JAX uses Channel as last dim

        # Convert inputs and labels to JAX arrays
        inputs = jnp.array(inputs.numpy())
        labels = jnp.array(labels.numpy())

        # Place inputs and labels explicitly on CPU
        inputs = jax.device_put(inputs, cpu_device)
        labels = jax.device_put(labels, cpu_device)

        # Compute metrics
        logits = model.apply(params, inputs)
        predicted = jnp.argmax(logits, axis=1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    jax_task()
