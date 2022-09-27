import librosa
import numpy as np

from app.config import settings


general_path = settings.resource_path

# Importing 1 file
y, sr = librosa.load(f"{general_path}/genres_original/reggae/reggae.00036.wav")

print("y:", y, "\n")
print("y shape:", np.shape(y), "\n")
print("Sample Rate (KHz):", sr, "\n")

# # Verify length of the audio
print("Check Len of Audio:", 661794 / 22050)
