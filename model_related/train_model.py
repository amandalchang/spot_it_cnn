import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import csv

from model import transform, CNN1, CLASS_LABEL_DICT

batch_size_input = 32
model_weights_path = "test_new_file"
train_loss_csv = "train_loss.csv"
test_loss_csv = "test_loss.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Icons(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(
            data_dir, transform=transform
        )  # Sets up the dataset using ImageFolder and applies any specified transformations.

    def __len__(self):
        return len(self.data)  # Returns the number of images in the dataset.

    def __getitem__(self, idx):
        return self.data[idx]  # Retrieves the image and label for a given index.

    @property  # @property: Turns a method into a read-only attribute.
    def classes(self):
        return self.data.classes  # Provides the list of class names.


symbol_list = [
    "anchor",
    "apple",
    "bird",
    "bomb",
    "bottle",
    "cactus",
    "candle",
    "car",
    "carrot",
    "cat",
    "cheese",
    "clock",
    "clover",
    "clown",
    "daisy",
    "dinosaur",
    "do_not_enter",
    "dog",
    "dolphin",
    "dragon",
    "droplet",
    "exclamation",
    "eye",
    "fire",
    "ghost",
    "hand",
    "heart",
    "ice",
    "igloo",
    "key",
    "knight",
    "ladybug",
    "leaf",
    "lightbulb",
    "lightning",
    "lips",
    "lock",
    "mallet",
    "man",
    "moon",
    "pencil",
    "question",
    "scissors",
    "skull",
    "snowflake",
    "snowman",
    "spider",
    "splat",
    "sun",
    "sunglasses",
    "target",
    "treble",
    "tree",
    "turtle",
    "web",
    "yin_yang",
    "zebra",
]

## Load training/test/val data
### transform defined in model.py
training_data = Icons(join("data", "train"), transform)
test_data = Icons(join("data", "test"), transform)
val_data = Icons(join("data", "validation"), transform)


train_loader = DataLoader(training_data, batch_size=batch_size_input, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size_input, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size_input, shuffle=False)

## Define model
model = CNN1(num_classes=len(CLASS_LABEL_DICT))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
# Lists to store training and test losses
train_losses = []
test_losses = []

num_epochs = 50


def train():
    # Training loop
    for _ in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Loop through training data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(
                device
            )  # Move inputs and labels to the same device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        running_test_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  #
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

        # Average test loss for the epoch
        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

    # Save the model's state_dict
    torch.save(model.state_dict(), model_weights_path)

    # Write the data to the CSV files for easy use after model is trained
    with open(train_loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for item in train_losses:
            writer.writerow([item])

    with open(test_loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for item in test_losses:
            writer.writerow([item])
