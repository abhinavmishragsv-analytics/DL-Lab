import os
import csv
import time

import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "tensorflow",
    "model",
    "mnist_tensorflow.keras"
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
# Load model
# ============================================================

print("=" * 60)
print("TENSORFLOW INFERENCE BENCHMARK")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# Load benchmark images
# ============================================================

print("\nLoading benchmark images...")

images = np.load(
    IMAGE_ARRAY_PATH
)

images = images.astype(
    "float32"
) / 255.0

print(f"Input shape: {images.shape}")


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

for _ in range(WARMUP_RUNS):

    model(
    images,
    training=False
)

# ============================================================
# Benchmark
# ============================================================

print(
    f"Running {BENCHMARK_RUNS} benchmark runs..."
)

times = []

predictions = None

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    probabilities = model(
    images,
    training=False
    ).numpy()
    end = time.perf_counter()

    times.append(
        end - start
    )

    predictions = probabilities


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
# Calculate accuracy
# ============================================================

predicted_labels = np.argmax(
    predictions,
    axis=1
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
    f"Number of images       : {len(images)}"
)

print(
    f"Warm-up runs           : {WARMUP_RUNS}"
)

print(
    f"Benchmark runs         : {BENCHMARK_RUNS}"
)

print(
    f"Average batch time     : {average_time * 1000:.4f} ms"
)

print(
    f"Median batch time      : {median_time * 1000:.4f} ms"
)

print(
    f"Minimum batch time     : {minimum_time * 1000:.4f} ms"
)

print(
    f"Maximum batch time     : {maximum_time * 1000:.4f} ms"
)

print(
    f"Std deviation          : {std_time * 1000:.4f} ms"
)

print(
    f"Average time / image   : {time_per_image * 1000:.4f} ms"
)

print(
    f"Inference accuracy     : {accuracy * 100:.2f}%"
)

print("=" * 60)