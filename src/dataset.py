import numpy as np


def load_dataset(debug=False):

    return open("task8/training/TRAIN_FILE_SMALL.txt", "r") if debug
    else dataset = open("task8/training/TRAIN_FILE.txt", "r")


def read_dataset(dataset, output_dict, merge_classes=False):
    X_raw = []
    Y = []
    i = 0
    for line in dataset:
        if i == 0:
            X_raw.append(process_train(line))
        if i == 1:
            if args.merge_classes:
                Y.append(output_dict[line.strip().split("(")[0]])
            else:
                Y.append(output_dict[line.strip()])
        if i == 2 or i == 3:
            pass
        i += 1
        if i % 4 == 0:
            i = 0
    return np.asarray(X_raw), np.asarray(Y)