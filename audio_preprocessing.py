# Function to generate spectrogram as feature
from random import randint

import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_audio(file_path, target_sample_length):
    # load audio
    y, sr = librosa.load(file_path)

    # Pad the audio signal with zeros to achieve the target length
    y_padded = librosa.util.pad_center(y, size=target_sample_length)

    return y_padded, sr

def generate_spectrogram(file_path, target_sample_length = 220500):
    try:
        # load audio
        y, sr = load_audio(file_path, target_sample_length)

        # generate spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        return D
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def generate_mel_spectrogram(file_path, target_sample_length = 220500):
    try:
        # load audio
        y, sr = load_audio(file_path, target_sample_length)

        S = librosa.feature.melspectrogram(y=y,
                                           sr=sr,
                                           n_mels=128 * 2)
        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

        return S_db_mel

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def spectrogram_exporter(n = 10):
    """
    This is just a helper script to generate some spectrograms to manually inspect.
    """
    df = pd.read_csv('data/dataset.csv')

    for i, row in df.iterrows():
        if i > n:
            break

        audio_path = f"data/audioset_audios/{row['ytid']}_cut.mp3"

        spec = generate_spectrogram(audio_path)
        mel_spec = generate_mel_spectrogram(audio_path)

        if spec is None:
            continue

        # generate spectrogram
        fig, ax = plt.subplots(figsize=(10, 5))
        img = librosa.display.specshow(spec,
                                       x_axis='time',
                                       y_axis='log',
                                       ax=ax)
        ax.set_title(f"Spectrogram {row['ytid']} ({row['label']})", fontsize=20)
        fig.colorbar(img, ax=ax, format=f'%0.2f')
        plt.savefig(f"data/spectrograms/{row['ytid']}-{row['label']}_spec.png")
        plt.show()

        # generate mel-spectrogram
        fig, ax = plt.subplots(figsize=(10, 5))
        img = librosa.display.specshow(mel_spec,
                                       x_axis='time',
                                       y_axis='log',
                                       ax=ax)
        ax.set_title(f"Mel-Spectrogram {row['ytid']} ({row['label']})", fontsize=20)
        fig.colorbar(img, ax=ax, format=f'%0.2f')
        plt.savefig(f"data/spectrograms/{row['ytid']}-{row['label']}_mel-spec.png")
        plt.show()

def random_frequency_mask(spectrogram, n_freq_masks=1, freq_mask_max_width=10):

    # Apply frequency masks
    for _ in range(n_freq_masks):
        # Randomly choose the frequency mask parameters
        mask_width = randint(1, freq_mask_max_width + 1)
        mask_start = randint(0, spectrogram.shape[0] - mask_width + 1)

        # Apply the frequency mask
        spectrogram[mask_start:mask_start + mask_width, :] = 0

    return spectrogram

if __name__ == '__main__':

    spec = generate_spectrogram('data/audioset_audios/cmlCMTqhOvI_cut.mp3')
    print(spec.shape)

    spec_aug = random_frequency_mask(spec)
    print(spec_aug.shape)

    print(spec_aug == spec)





