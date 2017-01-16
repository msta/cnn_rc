import numpy as np
import re

def load_dataset(debug=False):

    if debug:
        return open("task8/training/TRAIN_FILE_SMALL.txt", "r")
    else:
        return open("task8/training/TRAIN_FILE.txt", "r") 

def get_text(line):
    # split on quotes and get the raw text
    stripped = re.sub("\d{0,5}", "", line, count=1).strip()
    return stripped[1:len(stripped)-2]

def get_id(line):
    return re.findall("\d{0,5}", line)[0]


def read_dataset(dataset, 
                 output_dict, 
                 merge_classes=False):
    X_raw = []
    Y = []
    i = 0
    for line in dataset:
        if i == 0:
            X_raw.append(get_text(line))
        if i == 1:
            if merge_classes:
                Y.append(output_dict[line.strip().split("(")[0]])
            else:
                Y.append(output_dict[line.strip()])
        if i == 2 or i == 3:
            pass
        i += 1
        if i % 4 == 0:
            i = 0
    return np.asarray(X_raw), np.asarray(Y)

def gen_dataset(dataset, 
                 output_dict, 
                 merge_classes=False):
    X_raw = []
    Y = []
    i = 0
    current_id = ""
    for line in dataset:
        if i == 0:
            current_id = get_id(line)
        if i == 1:
            Y.append(current_id + "\t" + line.strip())
        if i == 2 or i == 3:
            pass
        i += 1
        if i % 4 == 0:
            i = 0
    return Y


def read_testset(dataset,
                output_dict):

    data = []
    ids = []
    for line in dataset:
        data.append(get_text(line))
        ids.append(get_id(line))    
    return data, ids