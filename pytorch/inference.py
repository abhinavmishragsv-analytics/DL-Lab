import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pytorch",
    "model",
    "mnist_pytorch.pth"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "benchmark_images"
)

IMAGE_ARRAY_PATH = os.path.join(
    IMAGE_DIR,
    "benchmark_images.npy"
)

LABEL_PATH = os.path.join(
    IMAGE_DIR,
    "labels.csv"
)


# ============================================================
# Model definition
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
print("PYTORCH INFERENCE BENCHMARK")
print("=" * 60)

print(f"\nDevice: {DEVICE}")


# ============================================================
# Load model
# ============================================================

print("\nLoading model...")

model = MNISTModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Load benchmark images
# ============================================================

print("\nLoading benchmark images...")

images = np.load(
    IMAGE_ARRAY_PATH
)

images = (
    images.astype("float32")
    / 255.0
)

images = torch.tensor(
    images,
    dtype=torch.float32
).to(DEVICE)

print(
    f"Input shape: {tuple(images.shape)}"
)


# ============================================================
# Load labels
# ============================================================

labels = []

with open(
    LABEL_PATH,
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        labels.append(
            int(row["true_label"])
        )

labels = np.array(labels)


# ============================================================
# Warm-up
# ============================================================

print(
    f"\nRunning {WARMUP_RUNS} warm-up runs..."
)

with torch.inference_mode():

    for _ in range(WARMUP_RUNS):

        model(images)


# ============================================================
# Benchmark
# ============================================================

print(
    f"Running {BENCHMARK_RUNS} benchmark runs..."
)

times = []

outputs = None

with torch.inference_mode():

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        outputs = model(images)

        end = time.perf_counter()

        times.append(
            end - start
        )


# ============================================================
# Calculate statistics
# ============================================================

times = np.array(times)

average_time = np.mean(times)
median_time = np.median(times)
minimum_time = np.min(times)
maximum_time = np.max(times)
std_time = np.std(times)

time_per_image = (
    average_time / len(images)
)


# ============================================================
# Predictions
# ============================================================

predicted_labels = (
    outputs.argmax(
        dim=1
    )
    .cpu()
    .numpy()
)

accuracy = np.mean(
    predicted_labels == labels
)


# ============================================================
# Display predictions
# ============================================================

print("\nPredictions:")
print("-" * 40)

for i in range(len(images)):

    print(
        f"image_{i + 1:02d}.png | "
        f"True: {labels[i]} | "
        f"Predicted: {predicted_labels[i]}"
    )


# ============================================================
# Display benchmark results
# ============================================================

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(
    f"Number of images       : "
    f"{len(images)}"
)

print(
    f"Warm-up runs           : "
    f"{WARMUP_RUNS}"
)

print(
    f"Benchmark runs         : "
    f"{BENCHMARK_RUNS}"
)

print(
    f"Average batch time     : "
    f"{average_time * 1000:.4f} ms"
)

print(
    f"Median batch time      : "
    f"{median_time * 1000:.4f} ms"
)

print(
    f"Minimum batch time     : "
    f"{minimum_time * 1000:.4f} ms"
)

print(
    f"Maximum batch time     : "
    f"{maximum_time * 1000:.4f} ms"
)

print(
    f"Std deviation          : "
    f"{std_time * 1000:.4f} ms"
)

print(
    f"Average time / image   : "
    f"{time_per_image * 1000:.4f} ms"
)

print(
    f"Inference accuracy     : "
    f"{accuracy * 100:.2f}%"
)

print("=" * 60)