from app.data.gztan.gztan import GtzanDataset

gtzan = GtzanDataset()

# print(gtzan.list_files_info())
# gtzan.play_original_file(original_file_path=None)
# gtzan.show_sound_wave(original_file_path=None)
# gtzan.show_spectogram_from_dataset(image_file_path=None)
# gtzan.create_decibel_spectogram_from_sound_file(sound_file_path=None)
# gtzan.create_mel_spectogram_from_sound_file(sound_file_path=None)
gtzan.compare_created_to_read_spectogram(sound_file_path=None)
