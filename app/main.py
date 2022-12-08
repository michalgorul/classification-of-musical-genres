# from multiprocessing import Process
#
# from keras.callbacks import History
#
# from app.gztan.data.data import list_output_of_generators, get_train_data_generator, get_validation_data_generator
# from app.gztan.data.gztan import GtzanDataset
# from app.gztan.model.model import build_model
# from app.plotting.plot import show_training_and_validation_loss, show_training_and_validation_accuracy
#
# # disable_cuda()
#
# # print(gtzan.list_files_info())
# # gtzan.data_init()
# # gtzan.sanity_data_test()
# gtzan = GtzanDataset()
# # gtzan.make_3_sec_wavs()
# # gtzan.data_init(sec_3=True)
#
# # gtzan.data_init()
# # gtzan.sanity_data_test()
#
#
# # build_model()
#
# list_output_of_generators()
#
# #
# train_data = get_train_data_generator()
# val_data = get_validation_data_generator()
#
# train_10sec_dir = gtzan.directories_10sec.get("train_dir")
# train_3sec_dir = gtzan.directories_3sec.get("train_dir")
#
# model = build_model()
#
# history: History = model.fit(
#     train_data,
#     steps_per_epoch=train_data.samples / train_data.batch_size,
#     epochs=20,
#     validation_data=val_data,
#     validation_steps=val_data.samples / val_data.batch_size,
# )
#
# model.save("gztan/model/gztan.h5")
# #
# train_loss_values = history.history.get("loss")
# val_loss_values = history.history.get("val_loss")
# train_accuracy = history.history.get("categorical_accuracy")
# val_accuracy = history.history.get("val_categorical_accuracy")
# num_of_epochs = range(1, len(train_accuracy) + 1)
#
# print(f"categorical_accuracy max: {max(history.history.get('categorical_accuracy'))}")
# print(f"val_categorical_accuracy max: {max(history.history.get('val_categorical_accuracy'))}")
# print(f"loss min: {min(history.history.get('loss'))}")
# print(f"val_loss min: {min(history.history.get('val_loss'))}")
#
# show_training_and_validation_loss(
#     epochs=num_of_epochs, loss_values=train_loss_values, val_loss_values=val_loss_values
# )
#
# show_training_and_validation_accuracy(
#     epochs=num_of_epochs, acc=train_accuracy, val_acc=val_accuracy
# )
from app.fma.data.fma import fma

fma.subset()