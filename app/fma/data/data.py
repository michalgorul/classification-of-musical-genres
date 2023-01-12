from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

TARGET_SIZE = (200, 400)
INPUT_SHAPE = (200, 400, 4)
BATCH_SIZE = 128


def get_train_data_generator(train_dir) -> DirectoryIterator:
    print("Creating train data generator")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgba",
    )
    print()
    return train_generator


def get_validation_data_generator(val_dir) -> DirectoryIterator:
    print("Creating validation data generator")
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgba",
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
