import cv2
import imutils
import os
import matplotlib.pyplot as plt
import numpy as np

def contrast_enhancement(img):
  # converting to LAB color space
  lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l_channel, a, b = cv2.split(lab)

  # Applying CLAHE to L-channel
  # feel free to try different values for the limit and grid size:
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl = clahe.apply(l_channel)

  # merge the CLAHE enhanced L-channel with the a and b channel
  limg = cv2.merge((cl,a,b))

  # Converting image from LAB Color model to BGR color spcae
  enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

  # Stacking the original image with the enhanced image
  # np.hstack((img, enhanced_img))
  return enhanced_img

# Load directory
card_directory = "dobble_deck02_cards_55"
#for file in os.listdir(card_directory):
file = "card42_01.tif" # placeholder
image = cv2.imread(os.path.join(card_directory, file))
# Increase the contrast in the lightness color space
output = contrast_enhancement(image)
# Convert back to RGB for visualization
enhanced_RGB = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

gray_enhanced = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) # grayscale
edges_enhanced = cv2.Canny(gray_enhanced, 50, 150) # find edges

# find contours on contrast enhanced grayscale image
contours, hierarchy = cv2.findContours(edges_enhanced, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("length of input image: ", image.shape)
print("length of output: ", output.shape)
print("length of gray_enhanced: ", gray_enhanced.shape)
print("length of edges_enhanced: ", edges_enhanced.shape)
# cv2.drawContours(edges_enhanced, cnts, -1, (0,255,0), 2)
plt.figure()
plt.axis("off")
plt.title(f"Enhanced Verison of {file}")
plt.imshow(enhanced_RGB)
plt.show()
plt.figure()
plt.axis("off")
plt.imshow(edges_enhanced, cmap="gray")
plt.show()

print("Number of Contours found = " + str(len(contours)))

# Draw all contours on top of the provided image
# -1 signifies drawing all contours
contour_drawn_RGB = enhanced_RGB.copy()
cv2.drawContours(contour_drawn_RGB, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
plt.figure()
plt.axis("off")
plt.imshow(contour_drawn_RGB)
plt.show()