import os
import time

import numpy as np
import keras
from keras import layers


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
    "mnist_keras.keras"
)


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(42)
keras.utils.set_random_seed(42)


# ============================================================
# Load MNIST
# ============================================================

print("=" * 60)
print("KERAS MNIST TRAINING")
print("=" * 60)

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    keras.datasets.mnist.load_data()
)


# ============================================================
# Preprocessing
# ============================================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"Training data: {x_train.shape}")
print(f"Test data:     {x_test.shape}")


# ============================================================
# Build model
# ============================================================

model = keras.Sequential([
    keras.Input(shape=(28, 28)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# Compile
# ============================================================

optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# Display architecture
# ============================================================

print("\nModel architecture:\n")

model.summary()


# ============================================================
# Training
# ============================================================

print("\nStarting training...")

start_time = time.perf_counter()

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

training_time = (
    time.perf_counter() - start_time
)


# ============================================================
# Evaluation
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


# ============================================================
# Save model
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(MODEL_PATH)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Training time : {training_time:.2f} seconds"
)

print(
    f"Test loss     : {test_loss:.4f}"
)

print(
    f"Test accuracy : {test_accuracy * 100:.2f}%"
)

print("\nModel saved to:")
print(MODEL_PATH)

print("=" * 60)