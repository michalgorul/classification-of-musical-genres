from app.data.gztan.gztan import GtzanDataset

gtzan = GtzanDataset()

# print(gtzan.list_files_info())
# gtzan.play_original_file(original_file_path=None)
gtzan.show_sound_wave(original_file_path=None)
gtzan.show_spectogram(image_file_path=None)
