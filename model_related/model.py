import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

IMG_SIZE = 64
CLASS_LABEL_DICT = {
    0: "anchor",
    1: "apple",
    2: "bird",
    3: "bomb",
    4: "bottle",
    5: "cactus",
    6: "candle",
    7: "car",
    8: "carrot",
    9: "cat",
    10: "cheese",
    11: "clock",
    12: "clover",
    13: "clown",
    14: "daisy",
    15: "dinosaur",
    16: "do_not_enter",
    17: "dog",
    18: "dolphin",
    19: "dragon",
    20: "droplet",
    21: "exclamation",
    22: "eye",
    23: "fire",
    24: "ghost",
    25: "hand",
    26: "heart",
    27: "ice",
    28: "igloo",
    29: "key",
    30: "knight",
    31: "ladybug",
    32: "leaf",
    33: "lightbulb",
    34: "lightning",
    35: "lips",
    36: "lock",
    37: "mallet",
    38: "man",
    39: "moon",
    40: "pencil",
    41: "question",
    42: "scissors",
    43: "skull",
    44: "snowflake",
    45: "snowman",
    46: "spider",
    47: "splat",
    48: "sun",
    49: "sunglasses",
    50: "target",
    51: "treble",
    52: "tree",
    53: "turtle",
    54: "web",
    55: "yin_yang",
    56: "zebra",
}


class CNN1(nn.Module):
    # based off the GT/Oregon State Visual
    def __init__(self, num_classes=len(CLASS_LABEL_DICT)):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Defines a max pooling layer
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, num_classes)
        self.small_dropout = nn.Dropout(p=0.1)
        self.large_dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.small_dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.small_dropout(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.small_dropout(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.small_dropout(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.large_dropout(x)
        return x


transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)
