import re

card_indexing = {
    1: {1, 8, 15, 16, 17, 18, 19, 20},
    2: {1, 9, 21, 22, 23, 24, 25, 26},
    3: {1, 10, 27, 28, 29, 30, 31, 32},
    4: {1, 11, 33, 34, 35, 36, 37, 38},
    5: {1, 12, 39, 40, 41, 42, 43, 44},
    6: {1, 13, 45, 46, 47, 48, 49, 50},
    7: {1, 14, 51, 52, 53, 54, 55, 56},
    8: {2, 8, 26, 27, 36, 41, 46, 51},
    9: {2, 9, 15, 32, 37, 42, 47, 52},
    10: {2, 10, 16, 21, 38, 43, 48, 53},
    11: {2, 11, 17, 22, 28, 44, 49, 54},
    12: {2, 12, 18, 23, 29, 33, 50, 55},
    13: {2, 13, 19, 24, 30, 34, 39, 56},
    14: {2, 14, 20, 25, 31, 35, 40, 45},
    15: {3, 8, 23, 32, 34, 43, 45, 54},
    16: {3, 9, 18, 30, 38, 40, 49, 51},
    17: {3, 10, 15, 24, 35, 44, 46, 55},
    18: {3, 11, 19, 21, 31, 41, 50, 52},
    19: {3, 12, 16, 25, 28, 36, 47, 56},
    20: {3, 13, 20, 22, 27, 33, 42, 53},
    21: {3, 14, 17, 26, 29, 37, 39, 48},
    22: {4, 8, 22, 31, 38, 39, 47, 55},
    23: {4, 9, 19, 29, 36, 44, 45, 53},
    24: {4, 10, 17, 25, 34, 42, 50, 51},
    25: {4, 11, 15, 23, 27, 40, 48, 56},
    26: {4, 12, 20, 21, 30, 37, 46, 54},
    27: {4, 13, 18, 26, 28, 35, 43, 52},
    28: {4, 14, 16, 24, 32, 33, 41, 49},
    29: {5, 8, 25, 30, 33, 44, 48, 52},
    30: {5, 9, 16, 27, 35, 39, 50, 54},
    31: {5, 10, 18, 22, 37, 41, 45, 56},
    32: {5, 11, 20, 24, 29, 43, 47, 51},
    33: {5, 12, 15, 26, 31, 34, 49, 53},
    34: {5, 13, 17, 21, 32, 36, 40, 55},
    35: {5, 14, 19, 23, 28, 38, 42, 46},
    36: {6, 8, 24, 28, 37, 40, 50, 53},
    37: {6, 9, 17, 31, 33, 43, 46, 56},
    38: {6, 10, 20, 23, 36, 39, 49, 52},
    39: {6, 11, 16, 26, 30, 42, 45, 55},
    40: {6, 12, 19, 22, 32, 35, 48, 51},
    41: {6, 13, 15, 25, 29, 38, 41, 54},
    42: {6, 14, 18, 21, 27, 34, 44, 47},
    43: {7, 8, 21, 29, 35, 42, 49, 56},
    44: {7, 9, 20, 28, 34, 41, 48, 55},
    45: {7, 10, 19, 26, 33, 40, 47, 54},
    46: {7, 11, 18, 25, 32, 39, 46, 53},
    47: {7, 12, 17, 24, 27, 38, 45, 52},
    48: {7, 13, 16, 23, 31, 37, 44, 51},
    49: {7, 14, 15, 22, 30, 36, 43, 50},
    50: {1, 2, 3, 4, 5, 6, 7, 57},
    51: {20, 26, 32, 38, 44, 50, 56, 57},
    52: {8, 9, 10, 11, 12, 13, 14, 57},
    53: {18, 24, 31, 36, 42, 48, 54, 57},
    54: {15, 21, 28, 33, 39, 45, 51, 57},
    55: {17, 23, 30, 35, 41, 47, 53, 57},
    56: {16, 22, 29, 34, 40, 46, 52, 57},
    57: {19, 25, 27, 37, 43, 49, 55, 57}
}

symbol_mapping = {
    1: "do_not_enter",
    2: "target",
    3: "lightning",
    4: "clown",
    5: "sunglasses",
    6: "spider",
    7: "anchor",
    8: "tree",
    9: "web",
    10: "cat",
    11: "bomb",
    12: "lock",
    13: "hand",
    14: "apple",
    15: "ghost",
    16: "man",
    17: "car",
    18: "moon",
    19: "eye",
    20: "clock",
    21: "scissors",
    22: "dinosaur",
    23: "splat",
    24: "key",
    25: "dog",
    26: "pencil",
    27: "exclamation",
    28: "knight",
    29: "cactus",
    30: "dolphin",
    31: "candle",
    32: "zebra",
    33: "droplet",
    34: "leaf",
    35: "fire",
    36: "turtle",
    37: "skull",
    38: "lips",
    39: "heart",
    40: "question",
    41: "cheese",
    42: "sun",
    43: "ladybug",
    44: "snowflake",
    45: "treble",
    46: "ice",
    47: "igloo",
    48: "carrot",
    49: "lightbulb",
    50: "bird",
    51: "clover",
    52: "daisy",
    53: "bottle",
    54: "yin_yang",
    55: "mallet",
    56: "dragon",
    57: "snowman"
}

def extract_card_number(filename):
    """
    Extracts the card number from a filename in the format 'card**_' where * can
    be anything and the characters following the underscore are ignored

    The function identifies the digits immediately following the prefix 'card' 
    and before the first underscore in the filename. For example:
    - For 'card14_01.tif', it will return 14.
    - For 'card07_01.tif', it will return 7.

    Parameters:
        filename (str): The name of the file to extract the card number from. 
                        It should follow the format 'card**_"

    Returns:
        str: The extracted card number as an int

    Raises:
        FileNotFoundError: If the number couldn't be extracted
    """
    match = re.search(r"card(\d+)_", filename)
    if match:
        return int(match.group(1))  # number extracting
    else:
        raise FileNotFoundError(f"The file '{filename}' could not be indexed")

def check_matches(user_card_idx, shared_card_idx, model_card_idx):
    """
    Checks the ground truth of card matches based on the card index, returns the
    match between user/shared and model/shared as a string
    """
    user_card_match_idx = (card_indexing[user_card_idx]).intersection(card_indexing[shared_card_idx])
    model_card_match_idx = (card_indexing[model_card_idx]).intersection(card_indexing[shared_card_idx])
    return symbol_mapping[user_card_match_idx.pop()], symbol_mapping[model_card_match_idx.pop()]

if __name__ == "__main__":
    true_user_match, true_model_match = check_matches(1, 14, 9)
    print(true_user_match)
    print(true_model_match)