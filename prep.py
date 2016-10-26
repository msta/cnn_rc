import numpy as np
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Preprocessor():

    def __init__(self, texts, debug=False):
        self.test = False
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(texts)
        self.nom1_idx = self.tokenizer.word_index['e1']
        self.nom2_idx = self.tokenizer.word_index['e2']

        self.debug = debug

    def test_mode(self):
        self.test = True

    def sequence(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def find_n(self, sequences):
        n = -1
        for seq in sequences:
            nom1_head = -1
            nom2_head = -1
            nom2_tail = -1
            for token_idx, token in enumerate(seq):
                # we subtract -1 and -3 to calculate for moved indices when e1 and e2 are removed
                if token == self.nom1_idx:
                    if nom1_head == -1:
                        nom1_head = token_idx
                if token == self.nom2_idx:
                    if nom2_head == -1:
                        nom2_head = token_idx
                    else:
                        nom2_tail = token_idx
            #### N has + 1 because we want the length of the sentence with relation mentions, not internal distance
            nx = (nom2_tail - nom1_head) + 1
            n = nx if nx > n else n
        self.n = n
        print "found n ", self.n

    def nom_arrays(self, sequences):
        X_nom_heads = []
        X_nom_rel = []
        for seq in sequences:
            nom1_head = -1
            nom2_head = -1
            nom2_tail = -1
            for token_idx, token in enumerate(seq):
                # we subtract -1 and -3 to calculate for moved indices when e1 and e2 are removed
                if token == self.nom1_idx:
                    if nom1_head == -1:
                        nom1_head = token_idx
                if token == self.nom2_idx:
                    if nom2_head == -1:
                        nom2_head = token_idx
                    else:
                        nom2_tail = token_idx
            X_nom_rel.append((nom1_head, nom2_tail))
            X_nom_heads.append([nom1_head, nom2_head])

        return X_nom_rel, X_nom_heads

    def word_idx(self):
        return self.tokenizer.word_index

    def pad_and_heads(self, sequences, X_nom_rel, X_nom_heads):
        X_padded_1 = []
        for seq_idx, seq in enumerate(sequences):
            
            head, tail = X_nom_rel[seq_idx]
            h = head
            t = tail + 1
            n_rest = max(0, self.n - (t - h))
            switch = True
            while n_rest > 0 and (h > 0 or t < len(seq)):
                if switch and h > 0:
                    h -= 1
                    n_rest -= 1
                elif t < len(seq):
                    t += 1
                    n_rest -= 1
                switch = not switch

            X_padded_1.append( seq[h:t])
            old_head1, old_head2 = X_nom_heads[seq_idx]
            X_nom_heads[seq_idx] = old_head1 - h, old_head2 - h
        return pad_sequences(X_padded_1, maxlen=self.n)


    def x_nom_pos(self, X_pad, X_nom_heads):
        X_nom_pos1 = []
        X_nom_pos2 = []
        for seq_idx, seq in enumerate(X_pad):
            nom_pos1 = []
            nom_pos2 = []
            for token_idx, token in enumerate(seq):
                head1, head2 = X_nom_heads[seq_idx]
                nom_pos1 = (head1 - token_idx) + self.n - 1 
                nom_pos2 = (head2 - token_idx) + self.n - 1
                nom_pos1.append(nom_pos1)
                nom_pos2.append(nom_pos2)
            X_nom_pos1.append(nom_pos1)
            X_nom_pos2.append(nom_pos2)
        return np.array(X_nom_pos1), np.array(X_nom_pos2)

    def preprocess(self, X):

        sequences = self.sequence(X)
        word_index = self.tokenizer.word_index
        word_index_new = self.word_idx()
        
        if not self.test:
            self.find_n(sequences)
        
        X_nom_rel, X_nom_heads = self.nom_arrays(sequences)
        sequences 
        if self.debug:
            print "-" * 30
            print "Embeddings before cleaning :"
            for l in sequences:
                print l
            print "-" * 30
            print "Embeddings after cleaning :"
            for l in sequences:
                print l
                    
        print "Maximum nominal distance: " , self.n

        X_pad = self.pad_and_heads(sequences, X_nom_rel, X_nom_heads)
        X_nom_pos1, X_nom_pos2 = self.x_nom_pos(X_pad, X_nom_heads)         
        
        return X_pad, X_nom_pos1, X_nom_pos2