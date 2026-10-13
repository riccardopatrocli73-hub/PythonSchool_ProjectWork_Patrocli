""" Visualization of some EMNIST Dataser characters """

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt

from src.dataset import label_to_letter, load_emnist_letters

NUMBER_OF_SAMPLES = 24
NUMBER_OF_COLUMNS = 6
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "emnist_samples.png"


def main():
    # Dataset is downloaded if not present
    dataset = load_emnist_letters(
        train=True,
        download=True,
        fix_orientation=True,
        normalize=False,
    )

    # print("Raw minimum label:", dataset.targets.min().item())
    # print("Raw maximum label:", dataset.targets.max().item())
    print(f"Number of training images: {len(dataset)}")
    print(f"Single image dimension: {dataset[0][0].shape}")
    print(f"Number of classes: 26")

    # Random images selection
    random_generator = random.Random(RANDOM_SEED)
    sample_indices = random_generator.sample(range(len(dataset)), NUMBER_OF_SAMPLES)

    number_of_rows = math.ceil(NUMBER_OF_SAMPLES / NUMBER_OF_COLUMNS)

    figure, axes = plt.subplots(
        number_of_rows,
        NUMBER_OF_COLUMNS,
        figsize=(10, 7),
    )

    for axis, index in zip(axes.flat, sample_indices):
        image, label = dataset[index]
        letter = label_to_letter(label)

        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(f"{label}: {letter}")
        axis.axis("off")

    figure.suptitle("EMNIST Letters", fontsize=16)
    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")

    print(f"Image saved in: {OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
