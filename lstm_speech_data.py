import numpy as np
import os, librosa, math, json

audio_silence_treshold = 45

def dense_to_one_hot(labels_dense, num_classes=8):
    """Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels_dense]


def load_audio_data(file_path, mfcc_features=8, height=200):
    json_file_path = file_path + str(mfcc_features) + '-' + str(height) + '.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            json_data = file.read()
            return json.loads(json_data)

    # 16000 Hz = VoIP
    wave, sr = librosa.load(file_path, mono=True, sr=16000)
    wave_frag_offsets = librosa.effects.split(wave, top_db=audio_silence_treshold)

    results = []
    for offsets in wave_frag_offsets:
        wave_fragment = wave[offsets[0]:offsets[1]]
        mfcc = librosa.feature.mfcc(wave_fragment, sr, n_mfcc=mfcc_features)
        _, y_size = mfcc.shape

        splitted_mfcc = np.array_split(mfcc, math.ceil(y_size / height), axis=1)

        for short_mfcc in splitted_mfcc:
            short_mfcc = np.pad(short_mfcc, ((0, 0), (0, height - len(short_mfcc[0]))), mode='constant', constant_values=0)
            results.append(np.array(short_mfcc).tolist())

    json_dump = json.dumps(results, separators=(',', ':'))
    with open(json_file_path, 'w') as file:
        file.write(json_dump)

    return results


def load_dataset(dir_path):
    dataset = []
    for p in os.listdir(dir_path):
        dataset.append(os.path.join(dir_path, p))

    return dataset


def mfcc_get_batch(files, batch_size=10, mfcc_features=8, height=200):
    batch_features = []
    labels = []

    for wav in files:
        if not wav.endswith(".wav"):
            continue

        # Our data is labeled 01-... but labels should be an int starting at 0
        emotion = int(os.path.basename(wav).split('-')[0]) - 1
        label = dense_to_one_hot(emotion, num_classes=8)
        audio_data = load_audio_data(wav, mfcc_features=mfcc_features, height=height)

        for mfcc in audio_data:
            labels.append(label)
            batch_features.append(mfcc)

            if len(batch_features) % 100 == 0:
                print('Loading Data Progress:', len(batch_features), '/', batch_size)

            # Return early if the batch_size has been satisfied
            if len(batch_features) >= batch_size:
                return batch_features, labels

    print('NOT ENOUGH DATA TO SATISFY BATCH_SIZE')
    return batch_features, labels
