import numpy as np
import matplotlib.pyplot as plt
import pickle


[e_pairs, aug_e_pairs, preds, aug_texts] = pickle.load(open("antonym_ei.pkl", "rb"))

n_bins = 5


output_dict = {
            "Cause-Effect" : 0,
            "Instrument-Agency" : 1,
            "Product-Producer" : 2,
            "Content-Container" : 3,
            "Entity-Origin" : 4,
            "Entity-Destination" : 5,
            "Component-Whole" : 6,
            "Member-Collection" : 7,
            "Message-Topic" : 8,
            "Other" : 9
        }

output_list = [ k for k,v in output_dict.items() ]


bin_levels = [0.,.5,.7, .8,.9,.95,.96,.97,.98,.99, 1.0]

#bin_levels = [.8,.9,.95,.96,.97,.98,.99, 1.0]


def fuck2():

    data_shit = []

    for i, clz in enumerate(output_list):
        if i != 9:
            data = preds[:,i*2:i*2+2].flatten()
        else:
            data = preds[:,18]
        data_shit.append([d for d in data if d > 0.8])


    #c_axis.set_title(clz)
    plt.hist(data_shit, bins='auto', label=output_list)
    plt.tight_layout()
    plt.show()


clz_idx = []
for i, clz in enumerate(output_list):
    if i != 9:
        data = preds[:,i*2:i*2+2].flatten()
    else:
        data = preds[:,18]
    clz_idx.append([(i,d) for (i,d) in enumerate(data) if d > 0.8])


def fuck():
    fig, axes = plt.subplots(nrows=5, ncols=2)
    axes_flat = axes.flatten()

    data_shit = []


    for i, clz in enumerate(output_list):
        c_axis = axes_flat[i]
        if i != 9:
            data = preds[:,i*2:i*2+2].flatten()
        else:
            data = preds[:,18]
        print("pred for class" + str(i) + "   " + str(len([d for d in data if d > 0.8])))
        data_shit.append([d for d in data if d > 0.8])
        c_axis.hist(data_shit[i], bins='auto', color=np.random.rand(3,))
        c_axis.set_title(clz)

    plt.tight_layout()
    plt.show()

fuck()