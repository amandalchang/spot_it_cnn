import cv2
from PIL import Image
import numpy as np


def contrast_enhancement(img, clipLimit):
    """
    Converts input image into a contrast enhanced version by
    converting the image into the LAB color space, applying
    contrast limited adaptive histogram equalization with a
    clip limit parameter to the lightness channel of LAB. This
    effectively darkens the image while retaining the other data.

    Args:
      img: a 3D array with BGR image data
      clipLimit: a float that controls the level of contrast enhancement
    Returns:
      A contrast enhanced 3D array with RGB image data
    """
    # converting to LAB color space
    lab = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    enhanced_RGB = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return enhanced_RGB


def find_object_contours(img, sym_card_size_ratio, binary_threshold):
    """
    Takes a card image and finds the outermost contours above a certain area
    threshold excluding the card itself.

    Args:
      img: the image itself, as an openCV output. 3D array with 3 color channels
      sym_card_size_ratio: a float which controls the min size a symbol has to be
        to qualify as an object. for example, 1/20 would mean the symbol is 1/20th
        the size of the full card (this could be wrong lol)
      binary_threshold: an integer binary threshold for the conversion to a binary
        image from a grayscaled version of img
    Returns:
      card_mask: a 2D black and white numpy array, where 255 is white and 0 is black.
        intended to be the filled in outline of the card (basically a circle)
      masked_binary: a 2D black and white numpy array, where 255 is white and 0 is black.
        intended to contain the just the objects on a black background
      object_contours: a list of contours which exceed the area threshold and are the
        the direct children of the card outline
    """
    gray_enhanced = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray_enhanced, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # create a mask to exclude the card
    card_mask = (
        np.ones_like(gray_enhanced, dtype=np.uint8) * 255
    )  # Start with a white mask
    card_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(card_mask, [card_contour], -1, color=0, thickness=cv2.FILLED)

    # Here we calculate the min required symbol size for size thresholding. With 1/20 for deck 2,
    # this also removes the little lines coming off the bomb and some sections of the smaller splats
    threshold_area = cv2.contourArea(card_contour) * (sym_card_size_ratio**2)
    # this parameter is for disqualifying if length or width of a contour is over half the card diameter
    length_width_bound = int(np.sqrt(cv2.contourArea(card_contour)) * 0.50)

    # apply the mask to the binary image
    masked_binary = cv2.bitwise_not(binary, binary, mask=card_mask)
    masked_binary = cv2.bitwise_not(masked_binary)
    outer_contours, _ = cv2.findContours(
        masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # area and bounding box thresholding
    object_contours = []
    for contour in outer_contours:
        if cv2.contourArea(contour) > threshold_area:
            _, _, w, h = cv2.boundingRect(contour)
            if w < length_width_bound and h < length_width_bound:
                object_contours.append(contour)

    return [card_mask, masked_binary, object_contours]
