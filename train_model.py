import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset",  # Replace with the path to your dataset
    target_size=(64, 64),
    batch_size=4,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(64, 64),
    batch_size=4,
    class_mode='binary',
    subset='validation'
)

# Step 2: Model Creation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Step 3: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # Adjust as needed
)

# Step 5: Save the Model
model.save('door_model_cpu1.h5')

print("Training complete! Model saved as 'door_model_cpu1.h5'.")
