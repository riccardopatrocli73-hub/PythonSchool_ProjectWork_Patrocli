"""CNN for handwritten character recognition"""

import torch
from torch import nn

# PyTorch standard model that has to distinguish 26 letters
class CharacterNN(nn.Module):
    def __init__(self, number_of_classes=26):
        super().__init__()

        """
        ---------------------------------------------------
        Visual features of the characters. 
        Input has (1, 28, 28) dimension where:
            - 1  -> grayscale channel
            - 28 -> height pixels
            - 28 -> width pixels
        ---------------------------------------------------
        """
        self.features = nn.Sequential(
            # first convolution: 1 channel in 32 features
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # non-linearity addition
            nn.ReLU(),
            # spatial dimension halving
            nn.MaxPool2d(kernel_size=2),
            # second convolution: 32 feature in 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # final dimension in 7x7
            nn.MaxPool2d(kernel_size=2),
            # each image is represented as 64x7x7
        )

        # letters classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)
