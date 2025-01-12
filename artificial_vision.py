import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera, you can change it if needed

# Initialize variables for the previous frame
prev_frame = None

while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    # Convert the frame to grayscale for image difference calculation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform image difference with the previous frame
    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray_frame)

        _, thresholded_diff = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)

        # Find the maximum difference value and its position (argmax)
        max_diff_value = np.max(thresholded_diff)
        max_diff_position = np.argmax(thresholded_diff)

        # Convert the 1D position into 2D (x, y) coordinates
        max_diff_x, max_diff_y = np.unravel_index(max_diff_position, thresholded_diff.shape)

        # Draw a marker (e.g., a circle) at the position of the maximum difference
        cv2.putText(frame, "max is here", (max_diff_y, max_diff_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(frame, (max_diff_y, max_diff_x), 5, (0, 0, 255), -1)  # Red circle at max difference position

        # Display the current frame with the marker at the maximum difference position
        cv2.imshow('Maximum Difference Position', frame)

    # Update the previous frame
    prev_frame = gray_frame

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
