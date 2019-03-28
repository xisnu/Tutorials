import numpy as np


def load_soybean_data(filename, for_rnn=False):
    f = open(filename)
    line = f.readline()
    xs = []
    ys = []
    while line:
        items = line.strip("\n").split(",")
        targetname = items[-1]
        target_index = int(targetname[-1]) - 1
        target_1h = [0, 0, 0, 0]
        target_1h[target_index] = 1.0
        features = items[:-1]
        xs.append(features)
        ys.append(target_1h)
        line = f.readline()
    if (for_rnn):
        xs = np.expand_dims(xs, 2)
    else:
        xs = np.asarray(xs)
    ys = np.asarray(ys)
    print("X Shape ", xs.shape, " Y Shape ", ys.shape)
    return xs, ys


def load_timeseries_data(timeseries_csv, nbclass, class_label_from_0=False):
    f = open(timeseries_csv)
    line = f.readline()
    sequence_length = []
    labels = []
    features = []
    while line:
        info = line.strip("\n").split(",")
        label = info[0]
        label_one_hot = np.zeros([nbclass])
        if (class_label_from_0):
            label_index = int(label)
        else:
            label_index = int(label) - 1
        label_one_hot[label_index] = 1
        feat = info[1:]
        nbfeatures = len(feat)
        sequence_length.append(nbfeatures)
        labels.append(label_one_hot)
        features.append(feat)
        line = f.readline()
    max_length = max(sequence_length)
    min_length = min(sequence_length)
    print("Maximum sequence length: ", max_length, " minimum sequence length: ", min_length)
    return np.asarray(features), np.asarray(labels), sequence_length
