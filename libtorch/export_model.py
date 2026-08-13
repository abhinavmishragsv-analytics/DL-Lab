import os
import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PYTORCH_MODEL = os.path.join(
    BASE_DIR,
    "pytorch",
    "model",
    "mnist_pytorch.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "libtorch",
    "model"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "mnist_libtorch.pt"
)


# ============================================================
# Model
# ============================================================

class MNISTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(784, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 10)

        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# Main
# ============================================================

print("=" * 60)
print("LIBTORCH MODEL EXPORT")
print("=" * 60)

print()
print("Loading PyTorch model:")
print(PYTORCH_MODEL)


# ------------------------------------------------------------
# Create model
# ------------------------------------------------------------

model = MNISTModel()


# ------------------------------------------------------------
# Load trained parameters
# ------------------------------------------------------------

checkpoint = torch.load(
    PYTORCH_MODEL,
    map_location="cpu"
)


if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint

        )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


# ------------------------------------------------------------
# Example input
# ------------------------------------------------------------

example_input = torch.randn(
    1,
    28,
    28
)


# ------------------------------------------------------------
# TorchScript export
# ------------------------------------------------------------

print()
print("Creating TorchScript model...")


scripted_model = torch.jit.trace(
    model,
    example_input
)


scripted_model = torch.jit.freeze(
    scripted_model
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


scripted_model.save(
    OUTPUT_FILE
)


print()
print("Model exported successfully!")

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("=" * 60)