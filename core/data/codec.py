# load file as time series
import functools
import os

import audioread.exceptions
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import soundfile
import torch
import wandb
from sklearn import preprocessing

from constants import OUTPUT_DIR
from core.utils import PreprocessingConfig, get_file_name_from

min_max_scaler = (
    joblib.load("scaler.joblib")
    if os.path.exists("scaler.joblib")
    else sklearn.preprocessing.MinMaxScaler(feature_range=(0, 10))
)
standard_scaler = (
    joblib.load("std_scaler.joblib") if os.path.exists("std_scaler.joblib") else sklearn.preprocessing.StandardScaler()
)
robust_scaler = (
    joblib.load("robust_scaler.joblib")
    if os.path.exists("robust_scaler.joblib")
    else sklearn.preprocessing.StandardScaler()
)

scalers = {
    "min_max": min_max_scaler,
    "standard": standard_scaler,
    "robust": robust_scaler,
}


def get_scaler(s="standard"):
    return scalers[s]


def chain_functions(functions, X, config: PreprocessingConfig):
    return functools.reduce(lambda acc, func: None if acc is None or len(acc) == 0 else func(acc, config), functions, X)


def audio_to_spec(y, config: PreprocessingConfig):
    S = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    return S


import math

min_db = math.inf
max_db = -math.inf


def audio_to_mel_spec(ys, config: PreprocessingConfig):
    def helper(
        y,
    ):
        from librosa.feature.spectral import melspectrogram

        S = melspectrogram(y=y, sr=config.sr, n_fft=config.n_fft, n_mels=config.n_mels, dtype=np.float32)
        shape = S.shape
        S = S[: min(config.input_height, shape[0]), : min(config.input_width, shape[1])]
        S_db = librosa.power_to_db(S, ref=np.max)
        # y = librosa.util.normalize(S_db)
        y = S_db
        # y = y * 2 + 1.

        # TODO: should be globally scaled
        # scale all bins together
        flattened = y.flatten().reshape(-1, 1)
        get_scaler(config.scale_method).partial_fit(flattened)
        y = get_scaler(config.scale_method).transform(flattened).reshape(y.shape)
        # global min_db, max_db
        # min_db = min(min_db, np.min(y))
        # max_db = max(max_db, np.max(y))
        # print(f'min: {min_db}')
        # print(f'max: {max_db}')
        # if not os.path.exists('scaler.joblib'):
        #     import joblib
        #     joblib.dump(min_max_scaler, 'scaler.joblib')
        # add an axis for one-channel
        y = y[np.newaxis, :]

        # feature_to_image(y.squeeze(), title=f'{get_file_name_from(y[0, 0, :100].tolist())}')
        # log_S = librosa.power_to_db(S, ref=np.max)
        return y

    return [helper(y) for y in ys]


def load_mel(ys, config: PreprocessingConfig):
    S = librosa.feature.melspectrogram(y=y, sr=config.sr, n_fft=config.n_fft, n_mels=config.n_mels, dtype=np.float32)

    return [helper(y) for y in ys]


def mel_to_audio(S, config: PreprocessingConfig):
    import soundfile as sf

    # S = S.astype(np.float32)
    # print(np.isnan(S).any())
    # Convert Mel-spectrogram to audio
    from librosa.feature.inverse import mel_to_audio

    # import joblib
    # scaler = joblib.load('scaler.joblib')
    scaler = get_scaler()
    S = scaler.inverse_transform(S)
    S_power = librosa.db_to_power(S)
    y_reconstructed = mel_to_audio(S_power, sr=config.sr)

    # Save the reconstructed audio
    file_name = f"{get_file_name_from(y_reconstructed)}.mp3"
    path = OUTPUT_DIR / file_name
    print(f"\nsaving to {path}")
    sf.write(file=path, data=y_reconstructed, samplerate=int(config.sr))
    return y_reconstructed


def load_audio(path, config: PreprocessingConfig, keep_channel=False):
    # ext = os.path.splitext(p)
    try:
        y, sr = librosa.load(
            path,
            sr=config.sr,
            mono=config.mono,
        )
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
        return sub_samples
    except soundfile.LibsndfileError as e:
        print(e)
        return None
    except audioread.exceptions.NoBackendError as e:
        print(e)
        return None
    except IsADirectoryError as e:
        print(e)
        return None
    except EOFError as e:
        print(e)
        return None


def feature_to_image(log_S, config: PreprocessingConfig = PreprocessingConfig(), show=False, title=""):
    if isinstance(log_S, torch.Tensor) and log_S.device != "cpu":
        log_S = log_S.cpu().float().numpy()
    show_spec([log_S], config, show, [title])


def show_spec(log_Ss, config: PreprocessingConfig = PreprocessingConfig(), show=False, titles=[], title=""):
    size = len(log_Ss)
    nrow = (size - 1) // 2 + 1
    fig, axes = plt.subplots(nrow, 2)
    plt.figure(figsize=(12 * 2, 6), dpi=800)

    if not titles:
        titles = [""] * size
    for log_S, ax, title in zip(log_Ss, axes, titles):
        if not title:
            title = get_file_name_from(log_S)
        librosa.display.specshow(log_S, sr=config.sr, x_axis="time", y_axis="mel", fmax=8000, ax=ax)
        ax.set_title(f"Mel Spectrogram of: {title}")
        # ax.colorbar(format='%+02.0f dB')

    if not title:
        title = get_file_name_from(log_Ss)
    print(f"\nsaving to {title}")
    fig.savefig(f"{title}.png", format="png")

    if show:
        fig.show()

    from io import BytesIO

    # 将Matplotlib绘制的图像保存到内存中的字节缓冲区
    buf = BytesIO()
    fig.savefig(fname=buf, format="png")
    # plt.close(fig)
    buf.seek(0)

    from PIL import Image

    # Use Pillow to open the image
    image = Image.open(buf)
    wandb.log({title: wandb.Image(image)})

    # image.save(fp=)
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
    def __init__(self, encodes, decodes, config: PreprocessingConfig):
        self.config = config
        self.encode = lambda x: chain_functions(encodes, x, config)
        self.decode = lambda x: chain_functions(decodes, x, config)

    def encode(self, X):
        return self.encode(X)

    def decode(self, X):
        return self.encode(X)


def build_codec(name, config):
    codecs = {
        "audio": Codec([lambda a, b: load_audio(a, b, keep_channel=True)], [], config=config),
        "mel": Codec([load_audio, audio_to_mel_spec], [mel_to_audio], config=config),
        "mel_image": Codec([load_mel], [mel_to_audio], config=config),
    }
    return codecs[name]


if __name__ == "__main__":
    from core.data.remixer import read_remixer_dataset
    from core.utils import TrainingConfig
    import os

    (
        files,
        ids,
        genres,
    ) = read_remixer_dataset("fma")
    codec = build_codec("audio", TrainingConfig().preprocessing)
    config = PreprocessingConfig()
    import soundfile as sf
    import threading

    seen = {}

    num_threads = 128

    def process(files, ids, genres):
        cnt = 0

        for file, _id, genre in zip(files, ids, genres):
            if "_" in file:
                continue
            Ss = codec.encode(file)
            if not Ss:
                continue
            for i, S in enumerate(Ss):
                if S.shape[1] == config.clipped_samples:
                    sf.write(
                        file=f"{file[:-4]}_{i}.mp3",
                        data=S.squeeze(),
                        samplerate=config.sr,
                    )
            cnt += 1
            if cnt % 100 == 0:
                print(f"{cnt} processed. current: {genre}")
            os.remove(file)

        global num_threads
        num_threads -= 1
        print(f"exiting {num_threads}")

    segment_size = len(files) // num_threads
    threads = []
    for i in range(num_threads):
        start = i * segment_size
        # 确保最后一个线程获取其余的所有元素
        end = None if i == num_threads - 1 else start + segment_size
        thread = threading.Thread(target=process, args=(files[start:end], ids[start:end], genres[start:end]))
        threads.append(thread)
        thread.start()
    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("done")
