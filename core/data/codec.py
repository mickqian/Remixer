# load file as time series
import functools

import audioread.exceptions
import librosa
import numpy as np
import soundfile
import soundfile as sf
import torch

from ..utils import PreprocessingConfig


def chain_functions(functions, X, config: PreprocessingConfig):
    return functools.reduce(lambda acc, func: None if not acc else func(acc, config), functions, X)


def audio_to_spec(y, config: PreprocessingConfig):
    S = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    return S


def audio_to_mel_spec(ys, config: PreprocessingConfig):
    def helper(y, ):
        S = librosa.feature.melspectrogram(y=y, sr=config.sr, n_fft=config.n_fft, n_mels=config.n_mels,
                                           dtype=np.float32)
        shape = S.shape
        S = S[:min(config.input_height, shape[0]), :shape[1] // 4 * 4]
        S_db = librosa.power_to_db(S, ref=np.max)
        # y = librosa.util.normalize(S_db)
        y = S_db
        y = y * 2 + 1.
        # add an axis for one-channel
        y = y[np.newaxis, :]

        feature_to_image(y, title=f'{hash(y)[1:6]}.png')

        # log_S = librosa.power_to_db(S, ref=np.max)
        return y

    return [helper(y) for y in ys]


def mel_to_audio(S, config: PreprocessingConfig):
    # S = S.astype(np.float32)
    # print(np.isnan(S).any())
    # Convert Mel-spectrogram to audio
    from librosa.feature.inverse import mel_to_audio
    y_reconstructed = mel_to_audio(S, sr=config.sr)

    # Save the reconstructed audio
    sf.write('reconstructed_audio.wav', y_reconstructed, int(config.sr))
    return y_reconstructed


def load_audio(path, config: PreprocessingConfig, keep_channel=False):
    # ext = os.path.splitext(p)
    try:
        y, sr = librosa.load(path, sr=config.sr, mono=config.mono, )
        if len(y.shape) == 2 and not keep_channel:
            y = y[0, :]
            y = y.squeeze()
        if keep_channel:
            if len(y.shape) > 1:
                y = y[:1, :]
            else:
                y = y[np.newaxis, :]
        length = y.shape[-1]
        if length < config.clipped_samples:
            print(y.shape)
        sub_array_size = config.clipped_samples
        sub_samples = np.split(y, range(sub_array_size, length, sub_array_size), axis=-1)
        # return sub_samples[:1]
        return sub_samples[:1]
    except soundfile.LibsndfileError as e:
        print(e)
        return None
    except audioread.exceptions.NoBackendError as e:
        print(e)
        return None
    except IsADirectoryError as e:
        print(e)
        return None


def feature_to_image(log_S, config: PreprocessingConfig = PreprocessingConfig(), show=False, title=''):
    import matplotlib.pyplot as plt
    if isinstance(log_S, torch.Tensor) and log_S.device != "cpu":
        log_S = log_S.cpu().float().numpy()
    plt.figure(figsize=(8, 6), dpi=800)
    librosa.display.specshow(log_S, sr=config.sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.title(f'Mel Spectrogram of: {title}')
    plt.colorbar(format='%+02.0f dB')
    if show:
        plt.show()

    from io import BytesIO
    # 将Matplotlib绘制的图像保存到内存中的字节缓冲区
    buf = BytesIO()
    plt.savefig(fname=buf, format='png')
    # plt.close(fig)
    buf.seek(0)
    plt.imsave(fname=title)

    from PIL import Image
    # Use Pillow to open the image
    image = Image.open(buf)
    # if show:
    #     import numpy
    #     image_array = numpy.array(image)
    #     plt.figure(dpi=400)
    #     plt.imshow(image_array)
    #     plt.axis('off')
    #     plt.title(f"mel spectrogram of: {title}")
    #     plt.show()
    return image


def db_spec_to_mel_spec(db_S, config: PreprocessingConfig):
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)


class Codec:
    def __init__(self, encodes, decodes, config):
        self.config = config
        self.encode = lambda x: chain_functions(encodes, x, config)
        self.decode = lambda x: chain_functions(decodes, x, config)

    def encode(self, X):
        return self.encode(X)

    def decode(self, X):
        return self.encode(X)


def build_codec(name, config):
    codecs = {
        'audio': Codec([lambda a, b: load_audio(a, b, keep_channel=True)], [], config=config),
        'mel': Codec([load_audio, audio_to_mel_spec], [mel_to_audio], config=config)
    }
    return codecs[name]
