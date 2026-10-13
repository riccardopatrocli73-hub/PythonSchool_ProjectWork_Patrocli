""" Prediction of a handwritten character from an image """

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from src.alphabet import LETTERS
from src.model import CharacterNN
from src.preprocessing import preprocess_character

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ch_cnn.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "preprocessed_character.png"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Recognize one handwritten character")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--top-k", type=int, default=3)

    return parser.parse_args()


def preprocess_image(image_path):
    """Backward-compatible alias for the shared preprocessing function."""

    return preprocess_character(image_path)


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = CharacterNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    arguments = parse_arguments()

    if not arguments.image.exists():
        raise FileNotFoundError(f"Image not found: {arguments.image}")

    if not 1 <= arguments.top_k <= 26:
        raise ValueError("--top-k must be between 1 and 26")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(arguments.checkpoint, device)

    image_tensor, processed_image = preprocess_character(arguments.image)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)

    top_probabilities, top_labels = torch.topk(
        probabilities,
        k=arguments.top_k,
        dim=1,
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    processed_image.save(arguments.output)

    print(f"Device: {device}")
    print(f"Processed image: {arguments.output}")
    print("\nPredictions:")

    for position, (probability, label) in enumerate(
        zip(
            top_probabilities[0].cpu(),
            top_labels[0].cpu(),
        ),
        start=1,
    ):
        print(f"{position}. {LETTERS[label.item()]}: " f"{probability.item():.2%}")


if __name__ == "__main__":
    main()
