import cv2
import numpy as np
# 1. Image Loading and Preprocessing
image_path = "normal.jpg"
image = cv2.imread(image_path)

# 2. Edge Detection
gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use adaptive thresholding for better edge detection
thresh = cv2.adaptiveThreshold(blurred, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 1)
kernel = np.ones((1, 1), np.uint8)

# Perform opening (erosion followed by dilation) to remove noise
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Perform closing (dilation followed by erosion) to close gaps in lines
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours in the image
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area to remove small artifacts
min_area = 4490  # Adjust this value based on the image resolution
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw the filtered contours on a blank image
output_image = np.zeros_like(image)
cv2.drawContours(output_image, filtered_contours, -1, (255, 255, 255), 2)
# 5. Simplification and Drawing
# Display or save the output image
cv2.imshow("Simplified Floor Plan", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the output
cv2.imwrite("simplified_floorplan.jpg", output_image)