# load file as time series
import functools

import librosa
import numpy as np
import soundfile as sf

from ..utils import PreprocessingConfig


def chain_functions(functions, X, config: PreprocessingConfig):
    return functools.reduce(lambda acc, func: func(acc, config), functions, X)


def audio_to_spec(y, config: PreprocessingConfig):
    S = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    return S


def audio_to_mel_spec(y, config: PreprocessingConfig):
    S = librosa.feature.melspectrogram(y=y, sr=config.sr, n_fft=config.n_fft, n_mels=config.n_mels, dtype=np.float32)
    # add an axis for one-channel
    shape = S.shape
    S = S[:min(config.input_height, shape[0]), :min(config.input_width, shape[1])]
    S = S[np.newaxis, :, :]
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def mel_to_audio(S, config: PreprocessingConfig):
    # S = S.astype(np.float32)
    print(np.isnan(S).any())
    # Convert Mel-spectrogram to audio
    from librosa.feature.inverse import mel_to_audio
    y_reconstructed = mel_to_audio(S, sr=config.sr)

    # Save the reconstructed audio
    sf.write('reconstructed_audio.wav', y_reconstructed, int(config.sr))
    return y_reconstructed


def load_audio(path, config: PreprocessingConfig):
    from pathlib import Path
    p = Path(path)
    # ext = os.path.splitext(p)
    y, sr = librosa.load(p)
    # y = y[:, :1290]
    # y = librosa.util.normalize(y)
    return y


class Codec:
    def __init__(self, encodes, decodes):
        config = PreprocessingConfig()
        self.encode = lambda x: chain_functions(encodes, x, config)
        self.decode = lambda x: chain_functions(decodes, x, config)

    def encode(self, X):
        return self.encode(X)

    def decode(self, X):
        return self.encode(X)


mel_codec = Codec([load_audio, audio_to_mel_spec], [mel_to_audio])
