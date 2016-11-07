import numpy as np
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Preprocessor():

    def __init__(self, texts, Y, debug=False, rand=False, clipping=False):
        self.test = False
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(texts)
        self.nom1_idx = self.tokenizer.word_index['e1']
        self.nom2_idx = self.tokenizer.word_index['e2']
        self.random_vecs = rand
        self.debug = debug
        self.clipping = clipping
        self.Y = Y
        self.ns = []

    def test_mode(self):
        self.test = True

    def sequence(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def find_n(self, sequences):
        if not self.clipping: 
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
                self.ns.append(nx)
                n = nx if nx > n else n
            self.n = n
        # testing clip
            import ipdb
            ipdb.sset_trace()
        else:
            print "testing clip"
            self.n = 18
        print "found n ", self.n


    def nominal_positions_and_clip(self, sequences):
        X_nom_heads = []
        X_nom_rel = []
        self.idx_to_keep = []
        for seq_idx, seq in enumerate(sequences):
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
            ### clipping experiment
            if self.clipping and nom2_head - nom1_head + 1 <= self.n:
                self.idx_to_keep.append(seq_idx)

            X_nom_rel.append((nom1_head, nom2_tail))
            X_nom_heads.append([nom1_head, nom2_head])
        if self.clipping:
            self.Y = np.asarray(self.Y)[self.idx_to_keep]
            one = np.asarray(X_nom_rel)[self.idx_to_keep]
            two =  np.asarray(X_nom_heads)[self.idx_to_keep] 
            three = np.asarray(sequences)[self.idx_to_keep]
            return one, two, three 

        else:
            return X_nom_rel, X_nom_heads, sequences

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
            tmp_nom_pos1 = []
            tmp_nom_pos2 = []
            for token_idx, token in enumerate(seq):
                head1, head2 = X_nom_heads[seq_idx]
                nom_pos1 = (head1 - token_idx) + self.n - 1 
                nom_pos2 = (head2 - token_idx) + self.n - 1
                tmp_nom_pos1.append(nom_pos1)
                tmp_nom_pos2.append(nom_pos2)
            X_nom_pos1.append(tmp_nom_pos1)
            X_nom_pos2.append(tmp_nom_pos2)
        return np.array(X_nom_pos1), np.array(X_nom_pos2)

    def reverse_sequence(self, seqs):
        inv_map = {v: k for k, v in self.tokenizer.word_index.iteritems()}
        return [[inv_map[s] for s in xs if s != 0] for xs in seqs]        


    def preprocess(self, X):


        sequences = self.sequence(X)
        word_index = self.tokenizer.word_index
        word_index_new = self.word_idx()
        
        if not self.test:
            self.find_n(sequences)
        
        X_nom_rel, X_nom_heads, sequences_clip = self.nominal_positions_and_clip(sequences)


        print "Maximum nominal distance: " , self.n

        X_pad = self.pad_and_heads(sequences_clip, X_nom_rel, X_nom_heads)
        X_nom_pos1, X_nom_pos2 = self.x_nom_pos(X_pad, X_nom_heads)         
        
        return X_pad, X_nom_pos1, X_nom_pos2, self.Y