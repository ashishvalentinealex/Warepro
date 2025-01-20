import shutil
import os

source_path = "/home/ashish/Warepro/test_door_model/dataset/frame_0.jpg"
destination_folder = "dataset/close"
os.makedirs(destination_folder, exist_ok=True)

for i in range(20):
    shutil.copy(source_path, f"{destination_folder}/close_{i}.jpg")
