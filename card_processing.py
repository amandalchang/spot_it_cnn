import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pytorch imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Image processing imports
import cv2
from PIL import Image

from image_processing import contrast_enhancement, find_object_contours
from model import transform, CLASS_LABEL_DICT


def get_icons(image, clip_limit, binary_threshold, sym_ratio, model):
    """
    Get the icons on a specific card

    image: an open cv image of a spot-it card
    clip_limit: a float that controls the level of contrast enhancement
    sym_card_size_ratio: a float which controls the min size a symbol has to be
        to qualify as an object. for example, 1/20 would mean the symbol is 1/20th
        the size of the full card (this could be wrong lol)
      binary_threshold: an integer binary threshold for the conversion to a binary
        image from a grayscaled version of img

    Returns:
        icon_names: A list of strings that represent all the icons on the card
        icon_contours: A list of contours which exceed the area threshold and are the
        the direct children of the card outline
    """
    enhanced_RGB = contrast_enhancement(image, clip_limit)

    # draw the smaller contours for visualization
    [card_mask, masked_binary, object_contours] = find_object_contours(
        enhanced_RGB.copy(), sym_ratio, binary_threshold
    )
    contoured_output_image = enhanced_RGB.copy()
    cv2.drawContours(
        contoured_output_image, object_contours, -1, (0, 255, 0), thickness=1
    )

    contour_areas = [cv2.contourArea(contour) for contour in object_contours]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    icons = []
    for i, contour in enumerate(object_contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Crop the ROI from the original image
        roi = enhanced_RGB[y : y + h, x : x + w]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour - [x, y]], -1, 255, thickness=cv2.FILLED)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
        # Convert to BGRA (add alpha channel)
        roi_bgra = cv2.cvtColor(roi_masked, cv2.COLOR_RGB2BGRA)
        roi_bgra[:, :, 3] = mask  # Set alpha channel based on the mask
        roi_rgb = cv2.cvtColor(roi_bgra, cv2.COLOR_BGRA2RGB)
        icons.append(roi_rgb)

    # model.load_state_dict(torch.load(model_path, weights_only=True))
    # model.eval()
    icon_names = []
    for icon in icons:
        image = Image.fromarray(icon).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Predict the label
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_label = torch.max(outputs, 1)
        icon_names.append(CLASS_LABEL_DICT[predicted_label.item()])
    return icon_names, object_contours


def display_card(image1, image2, card1_contour, card2_contour):
    """
    Display two cards side by side and highlight the associated contour

    Args:
        image1: an open cv image of a spot-it card
        image2: an open cv image of a  different spot-it card
        card1_contour: the contour to be highligted on image1
        card2_contour: the contour to be highlighted on image2
    """
    # Convert images to RGB for matplotlib
    image_rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    if card1_contour is not None and card2_contour is not None:
        # Draw contours directly on the images
        cv2.drawContours(image_rgb1, [card1_contour], -1, (0, 255, 0), thickness=2)
        cv2.drawContours(image_rgb2, [card2_contour], -1, (0, 255, 0), thickness=2)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display images
    axs[0].imshow(image_rgb1)
    axs[1].imshow(image_rgb2)

    # Turn off axis
    axs[0].axis("off")
    axs[1].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.show()
