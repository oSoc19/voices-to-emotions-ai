import os
import librosa
from random import shuffle
import numpy as np

data_dir = os.path.abspath('./data')

emotion_dict = {
    'W': 0,
    'L': 1,
    'E': 2,
    'A': 3,
    'F': 4,
    'T': 5,
    'N': 6
}


def dense_to_one_hot(labels_dense, num_classes=7):
    """Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels_dense]


def mfcc_batch_generator(batch_size=10):
    batch_features = []
    labels = []
    dataset_folder = os.path.join(data_dir, 'german')
    files = os.listdir(dataset_folder)

    while True:
        print("loaded batch of %d files" % len(files))
        shuffle(files)

        for wav in files:
            if not wav.endswith(".wav"):
                continue

            file_path = os.path.join(dataset_folder, wav)
            wave, sr = librosa.load(file_path, mono=True)
            wave_frag_offsets = librosa.effects.split(wave, top_db=35)

            for offsets in wave_frag_offsets:
                wave_fragment = wave[offsets[0]:offsets[1]]
                label = dense_to_one_hot(emotion_dict[wav[5:6]], 7)
                labels.append(label)
                mfcc = librosa.feature.mfcc(wave_fragment, sr)
                mfcc = np.pad(mfcc, ((0, 0), (0, 500 - len(mfcc[0]))), mode='constant', constant_values=0)
                batch_features.append(np.array(mfcc))
                if len(batch_features) >= batch_size:
                    yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                    batch_features = []  # Reset for next batch
                    labels = []
