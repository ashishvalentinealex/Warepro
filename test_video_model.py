from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Load the trained model
model = load_model('door_model_cpu1.h5')

# Define threshold for classification
threshold = 0.5

# Open the video file or camera
video_path = "/home/ashish/Downloads/1091169059-preview.mp4"  # Replace with your video file or '0' for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize the frame
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    normalized_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Predict the door status
    prediction = model.predict(normalized_frame)[0][0]
    status = "Open" if prediction > threshold else "Close"

    # Display the result on the frame
    cv2.putText(frame, f"Door Status: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame with the prediction
    cv2.imshow("Door Status", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
