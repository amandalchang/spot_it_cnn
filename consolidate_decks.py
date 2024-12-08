import os, shutil
from os import listdir
from os.path import join
import shutil

def consolidate_decks():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(join('data', 'icons')):
        os.makedirs(join('data', 'icons'))

    decks = ['deck-01', 'deck-02', 'deck-03', 'deck-04', 'deck-05', 'deck-06', 'deck-07', 'deck-08', 'deck-09']
    icons = ['anchor', 'apple', 'bird', 'bomb', 'bottle', 'cactus', 'candle', 'car', 'carrot', 'cat', 'cheese', 'clock', 'clover', 'clown', 'daisy', 'dinosaur', 'do_not_enter', 'dog', 'dolphin', 'dragon', 'droplet', 'exclamation', 'eye', 'fire', 'ghost', 'hand', 'heart', 'ice', 'igloo', 'key', 'knight', 'ladybug', 'leaf', 'lightbulb', 'lightning', 'lips', 'lock', 'mallet', 'man', 'moon', 'pencil', 'question', 'scissors', 'skull', 'snowflake', 'snowman', 'spider', 'splat', 'sun', 'sunglasses', 'target', 'treble', 'tree', 'turtle', 'web', 'yin_yang', 'zebra']

    for icon in icons:
        if not os.path.exists(join('data', 'icons', icon)):
            os.makedirs(join('data', 'icons', icon))

    for deck in decks:
        for icon in icons:
            images = listdir(join(deck, icon))
            for image in images:
                shutil.copy(join(deck, icon, image), join('data','icons', icon))
