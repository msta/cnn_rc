import json
import re
import os
import numpy as np
import nltk.data
import pickle

from ..semeval.prep import Preprocessor


class AcePrep(Preprocessor):

    def __init__(self, clipping_value=18):
        self.label_dict = {}
        self.word_dict = {}
        Preprocessor.__init__(self, clipping_value)

    def word_idx(self):
        return self.word_dict

    def load_dataset(self, train_file, debug=False):
        all_data = open("data/ace2005/training/ace2.csv").read().split("\n")
        self.word_dict = pickle.load(open("data/ace2005/training/vocab.pkl", "rb"))
        if debug:
            return all_data[:10000]
        return all_data[:-1]

    def read_dataset(self, train_file, debug=False):
        data = self.load_dataset(train_file, debug)

        input_tokens = []
        self.nom_heads = []
        labels = [] 

        for sample in data:
            splits = sample.split("\t")
            try:
                label_text = splits[-1]
                label = self.label_dict[label_text]
            except KeyError:
                self.label_dict[label_text] = len(self.label_dict)
                label = self.label_dict[label_text]
            head_1 = int(splits[1])
            head_2 = int(splits[2])
            ''' clipping '''
            if head_2 - head_1 + 1 > self.clipping_value:
                continue 
            labels.append(label)
            #labels.append(1)
            input_tokens.append(splits[0])
            self.nom_heads.append((head_1, head_2))

        return input_tokens, labels



    def fit_transform(self, data, labels):
        
        split_data = []
        for d in data:
            split_data.append([int(x) for x in d.split(" ")])
        
        """ must be already tokenized"""
        nom_arr_1, nom_arr_2 = self.create_nom_arrays(split_data, self.nom_heads)

        padded_data = self.fit_to_window(split_data, self.nom_heads)
        nom_arr_1 = self.fit_to_window(nom_arr_1, self.nom_heads)
        nom_arr_2 = self.fit_to_window(nom_arr_2, self.nom_heads)



        nom_arr_1 = self.normalize_nom_arr(nom_arr_1)
        nom_arr_2 = self.normalize_nom_arr(nom_arr_2)

        return (np.asarray(padded_data), 
            np.asarray(nom_arr_1), np.asarray(nom_arr_2), 
            {}, np.asarray([]), np.asarray([]), np.asarray(labels))

    def transform(self, data, labels):
        return self.fit_transform(data, labels)
        




































