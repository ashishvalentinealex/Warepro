from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Load the trained model
model = load_model('door_model_cpu1.h5')

# Define threshold for classification
threshold = 0.5

# Load and preprocess an image
image_path = "/home/ashish/Warepro/test_door_model/dataset/open/frame_330.jpgqq"  # Replace with the path to a test image
image = cv2.imread(image_path)

if image is None:
    print(f"Error loading image: {image_path}")
    exit()

resized_image = cv2.resize(image, (64, 64))
normalized_image = resized_image / 255.0
normalized_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

# Predict the door status
prediction = model.predict(normalized_image)[0][0]
status = "Open" if prediction > threshold else "Close"

# Display the result
cv2.putText(image, f"Door Status: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Door Status", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
