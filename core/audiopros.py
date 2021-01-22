
from defs import MEDIA_DIR

import librosa
import pydub

filename = librosa.example('nutcracker')

y, sr = librosa.load(filename)

tempo, beat_fraes = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

stram = librosa.stream(
                        filename,
                        block_length=256,
                        frame_length=4096,
                        hop_length=1024
                        )


