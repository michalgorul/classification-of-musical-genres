from app.gztan.data.gztan import GtzanDataset

# disable_cuda()

# print(gtzan.list_files_info())
# gtzan.data_init()
# gtzan.sanity_data_test()
gtzan = GtzanDataset()
# gtzan.make_3_sec_wavs()
gtzan.make_3_sec_images()

gtzan.data_init()
gtzan.sanity_data_test()


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
