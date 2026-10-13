"""
Fine-tune the character CNN from labeled local word images.

Image paths and expected text are read from a CSV manifest. Word images are
segmented with ``segment_characters``; images whose segment count does not match
the expected text are skipped so the training labels stay trustworthy.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.optim import Adam

from src.alphabet import LETTERS
from src.model import CharacterNN
from src.preprocessing import preprocess_character
from src.segment_word import segment_characters
from src.spell_correct import normalize_word

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "images_words.csv"
DEFAULT_BASE_CHECKPOINT = PROJECT_ROOT / "models" / "best_ch_cnn.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "best_ch_cnn_local_words.pt"


@dataclass(frozen=True)
class LabeledCharacter:
    tensor: torch.Tensor
    label: int
    source: str
    position: int


@dataclass(frozen=True)
class SkippedImage:
    image: str
    reason: str


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-tune the CNN using labeled images from the local project."
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV containing separate image and expected_text columns.",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter-pixels", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.025)
    parser.add_argument(
        "--include-single-characters",
        action="store_true",
        help="Also train from rows whose expected_text contains one letter.",
    )
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labeled_images(manifest_path, images_dir):
    with manifest_path.open(encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_columns = {"image", "expected_text"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing manifest columns: {missing_text}")

        labeled_images = []

        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            expected_text = (row.get("expected_text") or "").strip()

            if not image_value:
                raise ValueError(f"Empty image value in manifest row {row_number}")

            image_path = Path(image_value)

            if not image_path.is_absolute():
                image_path = images_dir / image_path

            labeled_images.append((image_path, expected_text))

    if not labeled_images:
        raise ValueError(f"The manifest contains no samples: {manifest_path}")

    return labeled_images


def collect_labeled_characters(
    images_dir,
    manifest_path,
    include_single_characters=False,
):
    samples = []
    skipped = []

    for image_path, manifest_text in load_labeled_images(manifest_path, images_dir):
        expected_text = normalize_word(manifest_text)

        if not expected_text:
            skipped.append(SkippedImage(image_path.name, "empty_normalized_label"))
            continue

        if len(expected_text) == 1 and not include_single_characters:
            skipped.append(SkippedImage(image_path.name, "single_character_disabled"))
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            skipped.append(SkippedImage(image_path.name, "unreadable_image"))
            continue

        if len(expected_text) == 1:
            try:
                tensor, _ = preprocess_character(image, trim_connectors=True)
            except Exception as error:
                skipped.append(SkippedImage(image_path.name, str(error)))
                continue

            samples.append(
                LabeledCharacter(
                    tensor=tensor.squeeze(0),
                    label=LETTERS.index(expected_text),
                    source=image_path.name,
                    position=1,
                )
            )
            continue

        try:
            _, segments = segment_characters(image)
        except Exception as error:
            skipped.append(SkippedImage(image_path.name, str(error)))
            continue

        if len(segments) != len(expected_text):
            skipped.append(
                SkippedImage(
                    image=image_path.name,
                    reason=(f"segments={len(segments)} expected={len(expected_text)}"),
                )
            )
            continue

        for position, (segment, letter) in enumerate(
            zip(segments, expected_text),
            start=1,
        ):
            left, top, right, bottom = segment.box
            crop = image[top:bottom, left:right]
            tensor, _ = preprocess_character(crop, trim_connectors=True)
            samples.append(
                LabeledCharacter(
                    tensor=tensor.squeeze(0),
                    label=LETTERS.index(letter),
                    source=image_path.name,
                    position=position,
                )
            )

    return samples, skipped


def _augment_batch(images, jitter_pixels, noise):
    if jitter_pixels > 0:
        shifted_images = []

        for image in images:
            vertical_shift = random.randint(-jitter_pixels, jitter_pixels)
            horizontal_shift = random.randint(-jitter_pixels, jitter_pixels)
            shifted_images.append(
                torch.roll(
                    image,
                    shifts=(vertical_shift, horizontal_shift),
                    dims=(-2, -1),
                )
            )

        images = torch.stack(shifted_images)

    if noise > 0:
        images = (images + torch.randn_like(images) * noise).clamp(-1, 1)

    return images


def train_local_model(
    samples,
    base_checkpoint,
    output_path,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    jitter_pixels,
    noise,
    device,
):
    checkpoint = torch.load(
        base_checkpoint,
        map_location=device,
        weights_only=True,
    )

    model = CharacterNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            images = torch.stack([sample.tensor for sample in batch_samples]).to(device)
            labels = torch.tensor(
                [sample.label for sample in batch_samples],
                dtype=torch.long,
                device=device,
            )

            if epoch <= max(1, int(epochs * 0.75)):
                images = _augment_batch(
                    images,
                    jitter_pixels=jitter_pixels,
                    noise=noise,
                )

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_samples)
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_samples += len(batch_samples)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss: {total_loss / total_samples:.4f} | "
            f"accuracy: {total_correct / total_samples:.2%}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(base_checkpoint),
            "local_samples": len(samples),
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        output_path,
    )


def main():
    arguments = parse_arguments()

    if not arguments.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {arguments.images_dir}")

    if not arguments.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {arguments.manifest}")

    if not arguments.base_checkpoint.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {arguments.base_checkpoint}"
        )

    if arguments.epochs < 1:
        raise ValueError("--epochs must be positive")

    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    set_random_seed(arguments.seed)

    samples, skipped = collect_labeled_characters(
        images_dir=arguments.images_dir,
        manifest_path=arguments.manifest,
        include_single_characters=arguments.include_single_characters,
    )

    if not samples:
        raise RuntimeError("No labeled character samples were collected")

    print(f"Collected character samples: {len(samples)}")

    if skipped:
        print("Skipped images:")
        for item in skipped:
            print(f"- {item.image}: {item.reason}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_local_model(
        samples=samples,
        base_checkpoint=arguments.base_checkpoint,
        output_path=arguments.output,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        jitter_pixels=arguments.jitter_pixels,
        noise=arguments.noise,
        device=device,
    )

    print(f"Saved fine-tuned checkpoint: {arguments.output}")


if __name__ == "__main__":
    main()
