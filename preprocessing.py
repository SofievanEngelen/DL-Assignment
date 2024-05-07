import os
import pickle
import numpy as np

import torch


def preprocess_dir(datadir):
    print("TEST")
    data = []

    if os.path.exists(datadir) and os.path.isdir(datadir):
        for file in os.listdir(datadir):
            # print(file)
            path = os.path.join(datadir, file)
            # print(path)
            with open(path, 'rb') as f:
                data.append(pickle.load(f))
        return data
