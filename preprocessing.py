import os
import pickle
import numpy as np

# import pytorch


def preprocess_dir(datadir):
    print("TEST")
    data = []

    for file in os.listdir(datadir):
        # print(file)
        path = os.path.join(datadir, file)
        # print(path)

        with open(path, 'rb') as f:
            data.append(pickle.load(f))

    print(data)
