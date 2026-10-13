""" Loading and preparation of EMNIST letters dataset """

import torch
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, random_split

from src.alphabet import LETTERS

# Project main folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class FixEMNISTOrientation:
    """ EMINST images rotation and reflection correction """

    def __call__(self, image):
        image = TF.rotate(image, -90)
        image = TF.hflip(image)
        return image


def zero_index_label(label):
    """
    EMINST uses labels from 1 to 26
    PyTorch requires labels from 0 to 25
    """
    return label - 1


def label_to_letter(label):
    """ numeric label to letter conversion """

    if label < 0 or label >= len(LETTERS):
        raise ValueError(f"Non-valid label: {label}")

    return LETTERS[label]


def build_transform(fix_orientation=True, normalize=False):
    """ Creation of images transformations """

    transformation_steps = []

    if fix_orientation:
        transformation_steps.append(FixEMNISTOrientation())

    transformation_steps.append(transforms.ToTensor())

    if normalize:
        transformation_steps.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    return transforms.Compose(transformation_steps)


def load_emnist_letters(
    train=True,
    download=True,
    fix_orientation=True,
    normalize=False,
):
    """ EMINST split letters loading """

    dataset = datasets.EMNIST(
        root=DATA_DIR,
        split="letters",
        train=train,        # training set if train = True, test set if train = False
        download=download,
        transform=build_transform(
            fix_orientation=fix_orientation,
            normalize=normalize,
        ),
        target_transform=zero_index_label,
    )

    return dataset


def create_dataloaders(
    batch_size=128,
    validation_fraction=0.10,
    seed=42,
    num_workers=0,
):
    """ Preparation of training, validation, test DataLoader """
    training_dataset = load_emnist_letters(
        train=True,
        download=True,
        fix_orientation=True,
        normalize=True,
    )

    test_dataset = load_emnist_letters(
        train=False,
        download=True,
        fix_orientation=True,
        normalize=True,
    )

    validation_size = int(len(training_dataset) * validation_fraction)  # 10% 
    training_size = len(training_dataset) - validation_size

    generator = torch.Generator().manual_seed(seed)

    training_subset, validation_subset = random_split(
        training_dataset,
        [training_size, validation_size],
        generator=generator,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    training_loader = DataLoader(
        training_subset,
        shuffle=True,
        **loader_options,
    )

    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )

    return training_loader, validation_loader, test_loader
