from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from app.gztan.data.gztan import gtzan

train_dir = gtzan.directories.get("train_dir")
val_dir = gtzan.directories.get("val_dir")


def get_train_data_generator() -> DirectoryIterator:
    """
    Yields batches of 150 ×150 RGB train images (shape (20, 150, 150, 3)) and categorical labels
    (shape(20,10)). There are 20 samples in each batch (the batch size). It yields these batches indefinitely:
    it loops endlessly over the images in the target folder. For this reason, break is used to end the iteration
    loop at some point
    :return: Train data generator of type DirectoryIterator
    """
    print("Creating train data generator")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
    )
    print()
    return train_generator


def get_validation_data_generator() -> DirectoryIterator:
    """
    Yields batches of 150 ×150 RGB validation images (shape (20, 150, 150, 3)) and categorical labels
    (shape(20,10)). There are 20 samples in each batch (the batch size). It yields these batches indefinitely:
    it loops endlessly over the images in the target folder. For this reason, break is used to end the iteration
    loop at some point
    :return: Validation data generator of type DirectoryIterator
    """
    print("Creating validation data generator")
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        val_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
    )
    print()
    return validation_generator


def list_output_of_generators() -> None:
    for data_batch, labels_batch in get_train_data_generator():
        print("Train generator:")
        print("\tData batch shape:", data_batch.shape)
        print("\tLabels batch shape:", labels_batch.shape)
        print()
        break

    for data_batch, labels_batch in get_validation_data_generator():
        print("Validation generator:")
        print("\tData batch shape:", data_batch.shape)
        print("\tLabels batch shape:", labels_batch.shape)
        break