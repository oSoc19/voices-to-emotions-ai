import os
import librosa
import numpy as np
import math

data_dir = os.path.abspath('./data')

emotion_dict = {
    'W': 0,  # angry
    'L': 1,  # calm
    'E': 2,  # disgust
    'A': 3,  # fearful
    'F': 4,  # happy
    'T': 5,  # sad
    'N': 6  # neutral
}


def dense_to_one_hot(labels_dense, num_classes=7):
    """Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels_dense]


def load_audio_data(file_path):
    wave, sr = librosa.load(file_path, mono=True)
    wave_frag_offsets = librosa.effects.split(wave, top_db=35)

    results = []
    for offsets in wave_frag_offsets:
        wave_fragment = wave[offsets[0]:offsets[1]]
        mfcc = librosa.feature.mfcc(wave_fragment, sr)
        _, y_size = mfcc.shape

        splitted_mfcc = np.array_split(mfcc, math.ceil(y_size / 500), axis=1)

        for short_mfcc in splitted_mfcc:
            short_mfcc = np.pad(short_mfcc, ((0, 0), (0, 500 - len(short_mfcc[0]))), mode='constant', constant_values=0)
            results.append(np.array(short_mfcc))

    return results


def load_dataset(dataset_folder=os.path.join(data_dir, 'german')):
    return os.listdir(dataset_folder)


def mfcc_get_batch(files, dataset_folder=os.path.join(data_dir, 'german'), batch_size=10):
    batch_features = []
    labels = []

    for wav in files:
        if not wav.endswith(".wav"):
            continue

        file_path = os.path.join(dataset_folder, wav)
        label = dense_to_one_hot(emotion_dict[wav[5:6]], 7)
        audio_data = load_audio_data(file_path)

        for mfcc in audio_data:
            labels.append(label)
            batch_features.append(mfcc)

            # Return early if the batch_size has been satisfied
            if len(batch_features) >= batch_size:
                return batch_features, labels

    print('NOT ENOUGH DATA TO SATISFY BATCH_SIZE')
    return batch_features, labels
