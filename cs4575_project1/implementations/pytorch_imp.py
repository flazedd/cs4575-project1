import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import cs4575_project1.implementations.constants as constants

def set_pytorch_seed(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def torch_task():
    # Set the seed for reproducibility
    set_pytorch_seed(42)

    # 1️⃣ Load & Preprocess MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0,), (1,))  # Normalize
    ])

    # Ensure the dataset is available; set download=True if needed
    train_dataset = datasets.MNIST(root='.\data\pytorch', train=True, download=False, transform=transform)
    test_dataset  = datasets.MNIST(root='.\data\pytorch', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2️⃣ Define CNN Model
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.relu2 = nn.ReLU()
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fc = nn.Linear(7 * 7 * 128, 10)  # 7*7*128 is the flattened size after pooling

        def forward(self, x):
            x = self.max_pool1(self.relu1(self.conv1(x)))
            x = self.max_pool2(self.relu2(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.fc(x)
            return x

    model = CNNModel()

    # 3️⃣ Compile the Model (define loss and optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Set device; for consistency, using CPU here
    device = torch.device('cpu')
    model.to(device)

    # 4️⃣ Train the Model
    epochs = constants.EPOCHS
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # 5️⃣ Evaluate the Model on Test Data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    torch_task()