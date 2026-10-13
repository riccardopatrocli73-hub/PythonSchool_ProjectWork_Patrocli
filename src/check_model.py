""" Check dataset and model compatibility before training """

from src.dataset import create_dataloaders
from src.model import CharacterNN

# input images -> [batch, 1, 28, 28]
# model output -> [batch, 26]
def main():
    training_loader, _, _ = create_dataloaders()

    images, labels = next(iter(training_loader))

    model = CharacterNN()   # first batch EMNIST passed to CharacterNN
    predictions = model(images)

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Model output:", predictions.shape)

    # The script terminates the execution if conditions not satisfied
    assert predictions.shape == (images.shape[0], 26)  
    assert labels.min() >= 0                           # values in 0-25
    assert labels.max() <= 25

    print("Dataset and model are compatible")


if __name__ == "__main__":
    main()
