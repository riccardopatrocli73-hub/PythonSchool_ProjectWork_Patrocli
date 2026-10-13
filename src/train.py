""" Training of EMNIST character recognition CNN """

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.dataset import create_dataloaders
from src.model import CharacterNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the character recognition CNN.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, dataloader, loss_function, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        # If training mode, compute gradients and update weights
        with torch.set_grad_enabled(is_training):
            predictions = model(images)
            loss = loss_function(predictions, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        predicted_labels = predictions.argmax(dim=1)  # Select class with highest logit

        total_loss += loss.item() * batch_size
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def main():
    arguments = parse_arguments()
    set_random_seed(arguments.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    training_loader, validation_loader, _ = create_dataloaders(
        batch_size=arguments.batch_size,
        seed=arguments.seed,
    )

    print(f"Training Samples: {len(training_loader.dataset)}")
    print(f"Validation Samples: {len(validation_loader.dataset)}")

    model = CharacterNN().to(device)
    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=arguments.learning_rate)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "best_ch_cnn.pt"

    best_validation_accuracy = 0.0

    for epoch in range(1, arguments.epochs + 1):
        training_loss, training_accuracy = run_epoch(
            model=model,
            dataloader=training_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy = run_epoch(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{arguments.epochs} | "
            f"train loss: {training_loss:.4f} | "
            f"train accuracy: {training_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

        # Model is stored if validation accuracy is increased
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )

            print(f"Best model saved: {checkpoint_path}")

    print(
        "Training completed. "
        f"Best validation accuracy: {best_validation_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
