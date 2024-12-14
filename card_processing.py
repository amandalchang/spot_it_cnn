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

# CLIP_LIMIT = 4.0
# BINARY_THRESHOLD = 200
# SYM_CARD_SIZE_RATIO = 1 / 20


def get_icons(image, clip_limit, binary_threshold, sym_ratio, model):
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
