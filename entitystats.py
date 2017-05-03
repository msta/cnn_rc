import numpy as np
import matplotlib.pyplot as plt
import pickle



def load_and_gen(data):


    def st(val):
        return set([frozenset(aug_e_pairs[i]) for i, e in enumerate(best_probs) if e > val])

    def gen():
        prob_lvls = np.arange(0.95, 1.0, 0.02)
        colors = "hello"
        x1 = []
        x2 = []

        for prob in prob_lvls:
            in_set = len(es.intersection(st(prob)))
            out_set = len(st(prob)) - in_set
            x1.append(in_set)
            x2.append(out_set)

        x = np.asarray([x1,x2]).transpose()
        plt.plot(prob_lvls, x1, label="entities in training set")
        plt.plot(prob_lvls, x2, label="new entities")
        plt.xlabel("$max_i^c P(y_i | x)$")
        plt.legend(prop={'size': 10})
        plt.show()

    if data == "antonym": 
        [e_pairs, aug_e_pairs, preds, aug_texts] = pickle.load(open("antonym_ei.pkl", "rb"))
    else:
        [e_pairs, aug_e_pairs, preds, aug_texts] = pickle.load(open("wordnet_ei.pkl", "rb"))


    best_probs = [np.max(prob) for prob in preds]
    print("number of samples :" + str(len(preds)))
    es = [frozenset(e) for e in e_pairs]

    gen()



def load_and_gen_distinct(data):


    def st(val):
        return [frozenset(aug_e_pairs[i]) for i, e in enumerate(best_probs) if e > val]

    def gen():
        prob_lvls = np.arange(0.90, 1.0, 0.02)
        colors = "hello"
        x1 = []
        x2 = []

        for prob in prob_lvls:
            in_set = len([e for e in st(prob) if e in es])
            out_set = len(st(prob)) - in_set
            x1.append(in_set)
            x2.append(out_set)

        x = np.asarray([x1,x2]).transpose()
        plt.plot(prob_lvls, x1, label="entities in training set")
        plt.plot(prob_lvls, x2, label="new entities")
        plt.xlabel("$max_i^c P(y_i | x)$")
        plt.legend(prop={'size': 10})
        plt.show()

    if data == "antonym": 
        [e_pairs, aug_e_pairs, preds, aug_texts] = pickle.load(open("antonym_ei.pkl", "rb"))
    else:
        [e_pairs, aug_e_pairs, preds, aug_texts] = pickle.load(open("wordnet_ei.pkl", "rb"))


    best_probs = [np.max(prob) for prob in preds]
    print("number of samples :" + str(len(preds)))
    es = [frozenset(e) for e in e_pairs]

    gen()



load_and_gen_distinct("wordnetw")
