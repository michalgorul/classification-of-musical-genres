from app.gztan.data.data import (
    get_train_data_generator,
    get_validation_data_generator,
)
from app.gztan.data.gztan import GtzanDataset
from app.gztan.model.model import build_model
from app.plotting.plot import (
    show_training_and_validation_loss,
    show_training_and_validation_accuracy,
)
from playground.utils.utils import disable_cuda

# disable_cuda()

# print(gtzan.list_files_info())
# gtzan.play_original_file(original_file_path=None)
# gtzan.show_sound_wave(original_file_path=None)
# gtzan.show_spectogram_from_dataset(image_file_path=None)
# gtzan.create_decibel_spectogram_from_sound_file(sound_file_path=None)
# gtzan.create_mel_spectogram_from_sound_file(sound_file_path=None)
# gtzan.compare_created_to_read_spectogram(sound_file_path=None)
# gtzan.data_init()
# gtzan.sanity_data_test()
gtzan = GtzanDataset()
gtzan.make_3_sec_wavs()
# build_model()

# list_output_of_generators()

# model = build_model()
#
# train_data = get_train_data_generator()
# val_data = get_validation_data_generator()
#
# history = model.fit(
#     train_data,
#     steps_per_epoch=train_data.samples / train_data.batch_size,
#     epochs=20,
#     validation_data=val_data,
#     validation_steps=val_data.samples / val_data.batch_size,
# )
#
# model.save("gztan/model/gztan.h5")
#
# train_loss_values = history.history.get("loss")
# val_loss_values = history.history.get("val_loss")
# train_accuracy = history.history.get("categorical_accuracy")
# val_accuracy = history.history.get("val_categorical_accuracy")
# num_of_epochs = range(1, len(train_accuracy) + 1)
#
# show_training_and_validation_loss(
#     epochs=num_of_epochs, loss_values=train_loss_values, val_loss_values=val_loss_values
# )
#
# show_training_and_validation_accuracy(
#     epochs=num_of_epochs, acc=train_accuracy, val_acc=val_accuracy
# )
