""" Evaluation of trained character recognition CNN """

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.alphabet import LETTERS
from src.dataset import create_dataloaders
from src.model import CharacterNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ch_cnn.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the CNN on the EMNIST test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def evaluate(model, dataloader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Rows = real letter, Columns = predicted letter
    confusion_matrix = torch.zeros(
        26,
        26,
        dtype=torch.int64,
    )

    errors = []

    # Inference
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(images)
            loss = loss_function(predictions, labels)

            predicted_labels = predictions.argmax(dim=1)
            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

            # Confusion matrix update
            indexes = labels * 26 + predicted_labels
            batch_matrix = torch.bincount(
                indexes,
                minlength=26 * 26,
            ).reshape(26, 26)

            confusion_matrix += batch_matrix.cpu()

            incorrect_indexes = (predicted_labels != labels).nonzero(as_tuple=True)[0]

            for index in incorrect_indexes:
                if len(errors) >= 24:
                    break

                errors.append(
                    (
                        images[index].cpu(),
                        labels[index].item(),
                        predicted_labels[index].item(),
                    )
                )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy, confusion_matrix, errors


def print_accuracy_per_letter(confusion_matrix):
    print("\nAccuracy per letter:")

    for label, letter in enumerate(LETTERS):
        total = confusion_matrix[label].sum().item()
        correct = confusion_matrix[label, label].item()
        accuracy = correct / total if total else 0.0

        print(f"{letter}: {accuracy:6.2%} " f"({correct}/{total})")


def save_confusion_matrix(confusion_matrix):
    figure, axis = plt.subplots(figsize=(11, 9))

    image = axis.imshow(
        confusion_matrix.numpy(),
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        title="EMNIST Letters - Confusion Matrix",
        xlabel="Predicted letter",
        ylabel="True letter",
        xticks=range(26),
        yticks=range(26),
        xticklabels=LETTERS,
        yticklabels=LETTERS,
    )

    figure.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Confusion matrix saved in: {output_path}")


def save_error_examples(errors):
    columns = 6
    rows = math.ceil(len(errors) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, rows * 2),
        squeeze=False,
    )

    for axis, (image, true_label, predicted_label) in zip(
        axes.flat,
        errors,
    ):
        # Images were normalized with mean=0.5 and std=0.5.
        image = image * 0.5 + 0.5

        axis.imshow(
            image.squeeze(0).clamp(0, 1),
            cmap="gray",
        )
        axis.set_title(
            f"True: {LETTERS[true_label]}\n" f"Predicted: {LETTERS[predicted_label]}"
        )
        axis.axis("off")

    for axis in axes.flat[len(errors) :]:
        axis.axis("off")

    figure.suptitle("Misclassified characters")
    figure.tight_layout()

    output_path = OUTPUT_DIR / "classification_errors.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Error examples saved in: {output_path}")


def print_most_common_confusions(confusion_matrix, limit=10):
    matrix = confusion_matrix.clone()
    matrix.fill_diagonal_(0)

    values, indexes = torch.topk(
        matrix.flatten(),
        k=limit,
    )

    print("\nMost common confusions:")

    for value, index in zip(values, indexes):
        true_label = index.item() // 26
        predicted_label = index.item() % 26

        print(
            f"{LETTERS[true_label]} -> "
            f"{LETTERS[predicted_label]}: "
            f"{value.item()} images"
        )


def main():
    arguments = parse_arguments()

    if not arguments.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {arguments.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = create_dataloaders(batch_size=arguments.batch_size)

    test_samples = len(test_loader.dataset)
    print(f"Test samples: {test_samples}")

    if test_samples != 20800:
        raise RuntimeError(
            "Unexpected test-set size. Check that the EMNIST "
            "test dataset is loaded with train=False."
        )

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model = CharacterNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Checkpoint validation accuracy: " f"{checkpoint['validation_accuracy']:.2%}")

    loss_function = nn.CrossEntropyLoss()

    test_loss, test_accuracy, confusion_matrix, errors = evaluate(
        model=model,
        dataloader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_accuracy_per_letter(confusion_matrix)
    print_most_common_confusions(confusion_matrix)
    save_confusion_matrix(confusion_matrix)
    save_error_examples(errors)


if __name__ == "__main__":
    main()
