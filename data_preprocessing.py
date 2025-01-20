from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(64, 64),
    batch_size=4,  # Small batch size for small dataset
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
