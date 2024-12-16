"""
Icons in each card of every deck are extracted via our image processing
These icons should be manually sorted by deck, so a folder for deck 1 has subfolder of each icon in it
The deck folders should be in the first level of the directory

These functions will consolidate these deck folders so that they are now all in a folder called `icons` in the `data` directory
This folder will still have a subfolder of each icon

The icons will then be split into train/test/validation datasets and be removed from the `icons` directory
"""

from consolidate_decks import consolidate_decks
from train_test_val_split import create_split

consolidate_decks()
create_split()
