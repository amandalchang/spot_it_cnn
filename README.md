# Overview
The purpose of this project is to train a model to recognize the icons on a Spoti-It! card in order to play the game Spot-It! The model is trained to recognizes the images on a Spot-It! card. In order to play the game, it creates a list of the icons on it recognizes on two cards, and finds the matching items in those lists. 

The game can be played in `play_game.ipynb`. The process and files related to creating training data and training the model are below. 

## Create Training Data
NOTE: You should not have to do any of this. `data.zip` already contains the datasets that you need to train the model yourself if you would like - this just details the process of creating that data.

The original dataset contains several decks from the game Spot-It! and can be found [here](https://www.kaggle.com/datasets/grouby/dobble-card-images/data). Because we want to train our model on the icons on the cards, and not the cards themselves, we have to do some image processing to extract the icons from the cards and save them. We use OpenCV to do this, and you can follow along with our process in `save_contours.ipynb`. 

In order to extract all the icons on a card, we increase the contrast of the image. We then use OpenCV to find the contours (which are the icons) on the card. The functions to do this can be found in `image_processing.py`. `save_contours.ipynb` extracts all the icons on all the cards in a given deck. However, it does not sort them. Once we extracted all the contours, we had to manually sort them ourselves. `sorter.py` helped with this process by "game-ifying" it, allowing us to view an icon and type where it should go. The data should be sorted within the decks, looking something likes this.
```

└───deck-1
│   │
│   └───anchor
│       │   anchor-1.png
│       │   anchor-2.png
│       │   ...
|   └───apple
|       |   apple-1.png
|       |   apple-2.png
|       |   ...
|   ...
│   
└───deck-2
│   │
│   └───anchor
│       │   anchor-1.png
│       │   anchor-2.png
│       │   ...
|   └───apple
|       |   apple-1.png
|       |   apple-2.png
|       |   ...
|   ...
```
Once the data has been sorted, run `create_data.py`. This consolidates all the decks into a `data\icons` directory, and then creates the train/test/validation split. The decks included in the data can be modified in `consolidate_decks.py`, and the train/test/validation split ratio can be controlled in `train_test_val_split.py.` All of these files are in the `data_organization` directory.

## Training the Model
More detail on training the model can be found in `Model Report.pdf`. Everything related to the model can be found in the directory `model_related`, including different model weights from prior iterations of the model. Functions for calculating the success of the model can be found in `model_metrics.py`. You can follow along with the process of training and evaluating the model in `model_testing.ipynb`, which is in the first level of the directory.

## Creating the Game