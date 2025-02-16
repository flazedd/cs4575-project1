import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

def set_pytorch_seed(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPUs, ensure CUDA has the same seed
    torch.backends.cudnn.deterministic = True  # Ensure deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable autotuner to ensure reproducibility


def torch_task():
    # Example usage:
    set_pytorch_seed(42)

    # 1️⃣ Load & Preprocess MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for better training stability
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # 2️⃣ Define CNN Model
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.max_pool1 = nn.MaxPool2d(2)

            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.relu2 = nn.ReLU()
            self.max_pool2 = nn.MaxPool2d(2)

            self.fc = nn.Linear(7 * 7 * 128, 10)  # 7*7*128 is the flattened size after pooling

        def forward(self, x):
            x = self.max_pool1(self.relu1(self.conv1(x)))
            x = self.max_pool2(self.relu2(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.fc(x)
            return x


    # Initialize the model
    model = CNNModel()

    # 3️⃣ Compile the Model (loss and optimizer in PyTorch)
    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters())  # Adam optimizer

    # 4️⃣ Train the Model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model.to(device)

    epochs = 5
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize the parameters

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # 5️⃣ Evaluate the Model on Test Data
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
