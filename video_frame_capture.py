import cv2
import os

video_path = "/home/ashish/Downloads/1091169059-preview.mp4"
output_folder = "dataset"
os.makedirs(f"{output_folder}/open", exist_ok=True)
os.makedirs(f"{output_folder}/close", exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 30  # Adjust as per your video's FPS
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_rate == 0:  # Save one frame per second
        cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)
    count += 1

cap.release()
