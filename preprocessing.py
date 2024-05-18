import os
import pickle

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_files(data_dir):
    data = []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(data_dir, file), 'rb') as f:
                data.append(pickle.load(f))

    return data


def extract_features(df, sr=8000):
    feature_vectors = []

    for utterance in df:
        y = utterance['audio_data']

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
        mfcc_delta_square = librosa.feature.delta(mfccs, order=2)
        mfcc_delta_square_mean = np.mean(mfcc_delta_square.T, axis=0)

        # Spectrogram (Mel-scaled)
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spec = librosa.power_to_db(spec, ref=np.max)
        log_spec_mean = np.mean(log_spec.T, axis=0)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # Combine all features into a single feature vector
        feature_vector = np.concatenate((mfcc_mean, mfcc_delta_mean, mfcc_delta_square_mean, log_spec_mean, [zcr_mean]))
        feature_vectors.append(feature_vector)

    return feature_vectors


def extract_labels(df):
    labels = []
    for utterance in df:
        labels.append(utterance['valence'])

    return labels


def preprocessDatasets(data_dir):
    if (os.path.exists("./train_dataset.pkl")) and (os.path.exists("./val_dataset.pkl")):
        print("Pre-loading Datasets...")
        with open("./pickles/train_dataset.pkl", 'rb') as f:
            train_dataset = pickle.load(f)
            print(f"Training dataset loaded from ./train_dataset.pkl")

        with open("./pickles/val_dataset.pkl", 'rb') as f:
            val_dataset = pickle.load(f)
            print(f"Validation dataset loaded from ./val_dataset.pkl")

    else:
        print("Processing files and creating Datasets...")
        data = load_files(data_dir)
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
        train_dataset = AudioDataset(train_data)
        val_dataset = AudioDataset(val_data)

        with open("./pickles/train_dataset.pkl", 'wb') as f:
            pickle.dump(train_dataset, f)
            print("Train dataset pickled")

        with open("./pickles/val_dataset.pkl", 'wb') as f:
            pickle.dump(val_dataset, f)
            print("Validation dataset pickled")

    return train_dataset, val_dataset


class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = extract_features(data)
        self.labels = extract_labels(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
