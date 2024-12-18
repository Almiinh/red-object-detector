"""
author: HUYNH Minh-Hoang

This script detect the movement by finding the max difference between frame and the previous frame.
"""

import cv2
import numpy as np
from skimage import morphology
from skimage import measure

def red_segmentation(frame):
    ### RED INTENSITY SEGMENTATION
    # Convert the frame to HSV for better detection of colors
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds of the red color in HSV space
    lower_red = np.array([0, 100, 95])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([175, 100, 95])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for the red color within the specified range
    red_mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Apply morphological operations to reduce noise in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    return red_mask

def trace_contours(frame, red_mask):
    ### CONTOURING
    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 600

    # Create a binary mask for the red objects
    binary_mask = np.zeros_like(red_mask)
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(binary_mask, [contour], -1, 255, -1)

    # Label the connected components in the binary mask
    labeled_objects = morphology.label(binary_mask, connectivity=2)

    # Draw 'red' text on the image where red objects are detected
    for region in measure.regionprops(labeled_objects):
        if region.area > min_contour_area:
            object_label = f'Red Object {region.label}'
            y, x, y_end, x_end = region.bbox
            cv2.putText(frame, object_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)

    return frame

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera, you can change it if needed
object_counter = 1
print("Press ESC to quit the frames")


while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break
    
    ### OPERATIONS
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = red_segmentation(frame)
    frame = trace_contours(frame, red_mask)

    ### DISPLAY
    cv2.imshow("HSV Colours", hsv_frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow('Red Object Detection', frame)

    # Exit when ESC is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
