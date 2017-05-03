import numpy as np
import math
import re
from collections import defaultdict
import logging

from ..functions import debug_print, debug_print_dict


from .classes import get_dict
from .tokenizer import SemevalTokenizer
from keras.preprocessing.sequence import pad_sequences

class Preprocessor():

    def __init__(self, 
        clipping_value=18, merge=False):

        self.n = clipping_value
        self.tokenizer = SemevalTokenizer()
        self.texts = None
        self.attentions_idx = {}
        self.Y = []
        self.clipping_value = clipping_value
        self.oov_val = -1
        self.nompos_normalizer = {}
        self.output_dict, self.reverse_dict = get_dict(merge)

    def remove_entities(self, texts):
        return np.asarray([re.sub("(<e1>|</e1>|<e2>|</e2>)*", "", text) for text in texts])

    def load_dataset(self, TRAIN_FILE):

        train_path = "data/semeval/training/" + TRAIN_FILE + ".txt"
        return open(train_path, "r") 

    def rreplace(self, s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    def get_text(self, line):
        # split on quotes and get the raw text
        stripped = re.sub("\d{0,5}", "", line, count=1).strip()
        stripped = re.sub("\"", "", stripped, count=1).strip()
        stripped = self.rreplace(stripped, "\"", "", 1)
        
        return stripped

    def get_id(self, line):
        return re.findall("\d{0,5}", line)[0]


    def read_dataset(self, TRAIN_FILE, debug=False, EXCLUDE_OTHER=False):

        dataset = self.load_dataset(TRAIN_FILE)

        X_raw = []
        Y = []
        i = 0
        for line in dataset:
            if i == 0:
                X_raw.append(self.get_text(line))
            if i == 1:
                if EXCLUDE_OTHER and line.strip() == 'Other':
                    X_raw = X_raw[:-1]
                else:
                    Y.append(self.output_dict[line.strip()])
            if i == 2 or i == 3:
                pass
            i += 1
            if i % 4 == 0:
                i = 0
        if debug:
            return np.asarray(X_raw)[:500], np.asarray(Y)[:500]
        else:
            return np.asarray(X_raw), np.asarray(Y)


    def read_semiset(self, aug_file, debug=False, EXCLUDE_OTHER=False, for_training=True, merge=False):

        dataset = self.load_dataset(aug_file)

        X_raw = []
        Y = []
        i = 0
        revers_ent = False
        flip_clz = []

        for line in dataset:
            if i == 0:
                text = self.get_text(line)
                e1_pos = text.find("e1")
                e2_pos = text.find("e2")
                if e2_pos < e1_pos:
                    revers_ent = True
                    text = text.replace("e1", "tmp").replace("e2", "e1").replace("tmp", "e2") 
                X_raw.append(text)
            if i == 1:
                if EXCLUDE_OTHER and line.strip() == 'Other':
                    X_raw = X_raw[:-1]
                elif for_training:
                    Y.append(self.output_dict[line.strip()])
                else:
                    pass
            if i == 2 or i == 3 or i == 4:
                pass
            i += 1
            if i % 5 == 0:
                i = 0
                flip_clz.append(revers_ent)
                revers_ent = False
        if len(X_raw) % 1000 == 0:
            print ("growing" + str(len(X_raw)))
        
        if debug:
            return np.asarray(X_raw)[:500], np.asarray(Y)[:500], None
        else:
            return X_raw, np.asarray(Y), np.asarray(flip_clz)


    def read_augset(self, aug_file, debug=False):

        dataset = self.load_dataset(aug_file)

        X_raw = []
        Y = []
        i = 0

        for line in dataset:
            if i == 0:
                text = self.get_text(line)
                X_raw.append(text)
            if i in  [1,2,3,4]:
                pass
            i += 1
            if i % 5 == 0:
                i = 0
        if len(X_raw) % 1000 == 0:
            print ("growing" + str(len(X_raw)))

        if debug:
            return np.asarray(X_raw)[:500]
        else:
            return X_raw

    def read_testset(self, dataset,
                    output_dict):
        data = []
        ids = []

        for line in dataset:
            data.append(self.get_text(line))
            ids.append(self.get_id(line))    
        return data, ids

  
    '''
    oov_val shows the number of words in the tokenizer
    with the first fitting
    '''
    def fit_tokenizer(self):
        #self.tokenizer = SemevalTokenizer()
        self.tokenizer.fit_on_texts(self.texts)

        self.oov_val = len(self.tokenizer.word_index)
    
    def normalize_nom_arr(self, arr):
        def normalize(tok):
            return tok + self.clipping_value - 1
            # if tok not in self.nompos_normalizer
            #     self.nompos_normalizer[tok] = len(self.nompos_normalizer)
            #     return self.nompos_normalizer[tok]
            # else:
            #     return self.nompos_normalizer[tok]
        return [normalize(tok) for tok in arr]


    def in_range(self, nom_1, nom_2):
        return abs(nom_1 - nom_2) < self.clipping_value

    def find_sent_outside_window(self, sequences, nom_heads):
        seq_to_keep = []
        for idx, seq in enumerate(sequences):
            nom_1, nom_2 = nom_heads[idx]
            if self.in_range(nom_1, nom_2):
                seq_to_keep.append(seq)

        return seq_to_keep
    
    ## used for testing only, with ids
    def transform(self, texts, ids, aux_texts=None):
        return self.fit_transform(texts, ids, fit=False, aux_texts=aux_texts)


    def fit_transform(self, texts, labels, fit=True, aux_texts=None):

        self.texts = texts
        
        logging.info("Tokenizing ...")
        if fit:
            self.fit_tokenizer()

        if aux_texts:
            self.tokenizer.fit_on_texts(aux_texts)
        
        self.Y = np.asarray(labels)

        logging.info("Sequencing everything ...")
        sequences, nominal_heads = zip(*self.tokenizer.sequence(texts))

        logging.info("Prepping the rest ...")
        e_pairs = self.get_e_pairs(sequences, nominal_heads)
        e_pairs_clip = self.find_sent_outside_window(e_pairs, nominal_heads)
#        texts = np.asarray(texts)
 #       texts_to_return = self.find_sent_outside_window(texts, nominal_heads) 
        sequences_clip = self.find_sent_outside_window(sequences, nominal_heads)
        self.Y = self.find_sent_outside_window(self.Y, nominal_heads)
        self.texts = self.find_sent_outside_window(np.asarray(texts), nominal_heads)

        nominal_heads_clip = [nom for nom in nominal_heads if self.in_range(nom[0], nom[1])]

        nominal_positions1, nominal_positions2 = (self.create_nom_arrays(sequences_clip, 
                                                    nominal_heads_clip))
        # Select sentence and fit to window so entities are in
        padded_sequences = self.fit_to_window(sequences_clip, nominal_heads_clip)

        nominal_positions1 = self.fit_to_window(nominal_positions1, nominal_heads_clip)
        nominal_positions2 = self.fit_to_window(nominal_positions2, nominal_heads_clip)



        nominal_positions1 = self.normalize_nom_arr(nominal_positions1)
        nominal_positions2 = self.normalize_nom_arr(nominal_positions2)

        return (padded_sequences, 
            np.asarray(nominal_positions1), np.asarray(nominal_positions2), 
            e_pairs_clip, None, None,
            self.Y)

    def fit_transform_aug(self, texts):

        self.texts = texts
        
        logging.info("Tokenizing ...")
        self.fit_tokenizer()


        logging.info("Sequencing everything ...")
        sequences, nominal_heads = zip(*self.tokenizer.sequence(texts))

        logging.info("Prepping the rest ...")
        e_pairs = self.get_e_pairs(sequences, nominal_heads)
        e_pairs_clip = self.find_sent_outside_window(e_pairs, nominal_heads)
#        texts = np.asarray(texts)
 #       texts_to_return = self.find_sent_outside_window(texts, nominal_heads) 
        sequences_clip = self.find_sent_outside_window(sequences, nominal_heads)
        self.texts = self.find_sent_outside_window(texts, nominal_heads)
        nominal_heads_clip = [nom for nom in nominal_heads if self.in_range(nom[0], nom[1])]

        nominal_positions1, nominal_positions2 = (self.create_nom_arrays(sequences_clip, 
                                                    nominal_heads_clip))
        # Select sentence and fit to window so entities are in
        padded_sequences = self.fit_to_window(sequences_clip, nominal_heads_clip)

        nominal_positions1 = self.fit_to_window(nominal_positions1, nominal_heads_clip)
        nominal_positions2 = self.fit_to_window(nominal_positions2, nominal_heads_clip)

        nominal_positions1 = self.normalize_nom_arr(nominal_positions1)
        nominal_positions2 = self.normalize_nom_arr(nominal_positions2)

        return (padded_sequences, 
            np.asarray(nominal_positions1), np.asarray(nominal_positions2), 
            e_pairs_clip, self.texts)

    def get_e_pairs(self, sequences, nominal_heads):
        reverse_seqs = self.reverse_sequence(sequences)
        return np.asarray([set([reverse_seqs[idx][one], reverse_seqs[idx][two]]) for idx, (one, two) in enumerate(nominal_heads)])

    def create_nom_arrays(self, sequences, nom_arrs):
        
        nom_arr_1 = []
        nom_arr_2 = []

        for seq_idx, seq in enumerate(sequences):
            tmp_1 = []
            tmp_2 = []
            head_val_1 = nom_arrs[seq_idx][0]
            head_val_2 = nom_arrs[seq_idx][1]
            for idx, tok in enumerate(seq):
                tmp_1.append((head_val_1 - idx)) 
                tmp_2.append((head_val_2 - idx)) 
            nom_arr_1.append(tmp_1)
            nom_arr_2.append(tmp_2)

        return nom_arr_1, nom_arr_2


    def get_nom_pos(self, text_split):
        pos1 = pos2 = -1
        inv_map = self.reverse_word_idx()
        for idx, token in enumerate(text_split):
            if token == 0:
                continue
            elif '<e1>' in inv_map[token]:
                pos1 = idx
            elif '<e2>' in inv_map[token]:
                pos2 = idx
        return pos1, pos2

    def reverse_word_idx(self):
        ret_dict = {v: k for k, v in self.word_idx().items()}
        ret_dict[0] = "PAD_TOK"
        return ret_dict

    def word_idx(self):
        return self.tokenizer.word_index

    ''' adjusts sequences so to include tokens up to clipping value '''
    def fit_to_window(self, sequences, nominal_heads):
        padded_sequence = []
        for seq_idx, seq in enumerate(sequences):
            head, tail = nominal_heads[seq_idx]
            h = head
            t = tail + 1
            n_rest = max(0, self.clipping_value - (t - h))
            while n_rest > 0 and (h > 0 or t < len(seq)):
                if h > 0:
                    h -= 1
                    n_rest -= 1
                elif t < len(seq):
                    t += 1
                    n_rest -= 1
            padded_sequence.append( seq[h:t])
        return pad_sequences(padded_sequence, self.clipping_value, value=0)


    def reverse_sequence(self, seqs):
        inv_map = self.reverse_word_idx()
        return [[inv_map[s] for s in xs] for xs in seqs]        



