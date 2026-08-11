import os
import csv
import struct
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "benchmark_images"
)

LABEL_FILE = os.path.join(
    IMAGE_DIR,
    "labels.csv"
)

NPY_FILE = os.path.join(
    IMAGE_DIR,
    "benchmark_images.npy"
)


# ============================================================
# Create directories
# ============================================================

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


# ============================================================
# Load MNIST
# ============================================================

print("Downloading/loading MNIST...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(
    f"Training images: {len(x_train)}"
)

print(
    f"Test images: {len(x_test)}"
)

print(
    f"Image shape: {x_train.shape[1:]}"
)


# ============================================================
# Select 10 fixed benchmark images
# ============================================================

indices = list(range(10))

benchmark_images = x_test[indices]
benchmark_labels = y_test[indices]


# ============================================================
# Save individual PNG images
# ============================================================

for i, image in enumerate(
    benchmark_images,
    start=1
):

    filename = f"image_{i:02d}.png"

    filepath = os.path.join(
        IMAGE_DIR,
        filename
    )

    Image.fromarray(image).save(
        filepath
    )


# ============================================================
# Save benchmark NumPy array
# ============================================================

np.save(
    NPY_FILE,
    benchmark_images
)


# ============================================================
# Save benchmark labels
# ============================================================

with open(
    LABEL_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["image", "true_label"]
    )

    for i, label in enumerate(
        benchmark_labels,
        start=1
    ):

        writer.writerow(
            [
                f"image_{i:02d}.png",
                int(label)
            ]
        )


# ============================================================
# Save MNIST in IDX binary format
# ============================================================

def save_images_idx(
    images,
    filepath
):

    with open(
        filepath,
        "wb"
    ) as file:

        # Magic number for images
        file.write(
            struct.pack(
                ">IIII",
                2051,
                images.shape[0],
                images.shape[1],
                images.shape[2]
            )
        )

        file.write(
            images.astype(
                np.uint8
            ).tobytes()
        )


def save_labels_idx(
    labels,
    filepath
):

    with open(
        filepath,
        "wb"
    ) as file:

        # Magic number for labels
        file.write(
            struct.pack(
                ">II",
                2049,
                labels.shape[0]
            )
        )

        file.write(
            labels.astype(
                np.uint8
            ).tobytes()
        )


# ============================================================
# Export training dataset
# ============================================================

train_images_file = os.path.join(
    DATASET_DIR,
    "train-images-idx3-ubyte"
)

train_labels_file = os.path.join(
    DATASET_DIR,
    "train-labels-idx1-ubyte"
)

test_images_file = os.path.join(
    DATASET_DIR,
    "test-images-idx3-ubyte"
)

test_labels_file = os.path.join(
    DATASET_DIR,
    "test-labels-idx1-ubyte"
)


save_images_idx(
    x_train,
    train_images_file
)

save_labels_idx(
    y_train,
    train_labels_file
)

save_images_idx(
    x_test,
    test_images_file
)

save_labels_idx(
    y_test,
    test_labels_file
)


# ============================================================
# Display results
# ============================================================

print("\nBenchmark images created:")
print("-" * 40)

for i, label in enumerate(
    benchmark_labels,
    start=1
):

    print(
        f"image_{i:02d}.png -> digit {label}"
    )


print("\nBenchmark files saved to:")
print(IMAGE_DIR)


print("\nBenchmark image array shape:")
print(benchmark_images.shape)


print("\nComplete MNIST dataset exported:")
print(
    train_images_file
)

print(
    train_labels_file
)

print(
    test_images_file
)

print(
    test_labels_file
)

print(
    "\nDataset preparation completed successfully!"
)