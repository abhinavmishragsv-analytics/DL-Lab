import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mnist_pytorch.pth"
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# Model
# ============================================================

class MNISTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                784,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                10
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Header
# ============================================================

print("=" * 60)
print("PYTORCH MNIST TRAINING")
print("=" * 60)

print(f"\nDevice: {DEVICE}")


# ============================================================
# Load MNIST
# ============================================================

print("\nLoading MNIST dataset...")

import tensorflow as tf

(
    (x_train, y_train),
    (x_test, y_test)
) = tf.keras.datasets.mnist.load_data()


# ============================================================
# Preprocessing
# ============================================================

x_train = (
    x_train.astype("float32")
    / 255.0
)

x_test = (
    x_test.astype("float32")
    / 255.0
)


# Convert NumPy → PyTorch tensors

x_train = torch.tensor(
    x_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

x_test = torch.tensor(
    x_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


print(
    f"Training data: {x_train.shape}"
)

print(
    f"Test data:     {x_test.shape}"
)


# ============================================================
# DataLoader
# ============================================================

train_dataset = TensorDataset(
    x_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# Create model
# ============================================================

model = MNISTModel().to(DEVICE)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Display architecture
# ============================================================

print("\nModel architecture:\n")

print(model)


# ============================================================
# Training
# ============================================================

print("\nStarting training...")

start_time = time.perf_counter()


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        # Backward pass

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Statistics

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy * 100:.2f}%"
    )


training_time = (
    time.perf_counter()
    - start_time
)


# ============================================================
# Evaluation
# ============================================================

print("\nEvaluating model...")

model.eval()

with torch.no_grad():

    test_outputs = model(
        x_test.to(DEVICE)
    )

    predictions = (
        test_outputs.argmax(dim=1)
    )

    test_accuracy = (
        (predictions == y_test.to(DEVICE))
        .float()
        .mean()
        .item()
    )


# ============================================================
# Save model
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Training time : "
    f"{training_time:.2f} seconds"
)

print(
    f"Test accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print("\nModel saved to:")
print(MODEL_PATH)

print("=" * 60)