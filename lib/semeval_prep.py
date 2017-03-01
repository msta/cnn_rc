import numpy as np
import math
import re
from collections import defaultdict
import logging

from .tokenizer import SemevalTokenizer
from .functions import debug_print, debug_print_dict
from keras.preprocessing.sequence import pad_sequences

class Preprocessor():

    def __init__(self, 
        clipping_value=18):

        self.n = clipping_value
        self.tokenizer = None
        self.texts = None
        self.attentions_idx = {}
        self.Y = []
        self.clipping_value = clipping_value
        self.oov_val = -1
        self.nompos_normalizer = {}
    '''
    oov_val shows the number of words in the tokenizer
    with the first fitting
    '''
    def fit_tokenizer(self):
        self.tokenizer = SemevalTokenizer()
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
                    Y.append(output_dict[line.strip()])
            if i == 2 or i == 3:
                pass
            i += 1
            if i % 4 == 0:
                i = 0
        if debug:
            return np.asarray(X_raw)[:5], np.asarray(Y)[:5]
        else:
            return np.asarray(X_raw), np.asarray(Y)


    def gen_dataset(self, dataset, 
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


    def read_testset(self, dataset,
                    output_dict):
        data = []
        ids = []

        for line in dataset:
            data.append(self.get_text(line))
            ids.append(self.get_id(line))    
        return data, ids

    ## used for testing only, with ids
    def transform(self, texts, ids):

        return self.fit_transform(texts, ids, fit=False)

    def in_range(self, nom_1, nom_2):
        return abs(nom_1 - nom_2) < self.clipping_value

    def fit_to_window(self, sequences, nom_heads):
        seq_to_keep = []
        for idx, seq in enumerate(sequences):
            nom_1, nom_2 = nom_heads[idx]
            if self.in_range(nom_1, nom_2):
                seq_to_keep.append(seq)

        return seq_to_keep


    def fit_transform(self, texts, labels, fit=True):

        self.texts = texts
        
        if fit:
            self.fit_tokenizer()
        

        self.Y = np.asarray(labels)

        sequences, nominal_heads = zip(*self.tokenizer.sequence(texts))

        sequences_clip = self.fit_to_window(sequences, nominal_heads)
        self.Y = self.fit_to_window(self.Y, nominal_heads)
        nominal_heads_clip = [nom for nom in nominal_heads if self.in_range(nom[0], nom[1])]

        nominal_positions1, nominal_positions2 = (self.create_nom_arrays(sequences_clip, 
                                                    nominal_heads_clip))
        # Select sentence and fit to window so entities are in
        padded_sequences = self.fit_to_window2(sequences_clip, nominal_heads_clip)

   


        nominal_positions1 = self.fit_to_window2(nominal_positions1, nominal_heads_clip)
        nominal_positions2 = self.fit_to_window2(nominal_positions2, nominal_heads_clip)

        nominal_positions1 = self.normalize_nom_arr(nominal_positions1)
        nominal_positions2 = self.normalize_nom_arr(nominal_positions2)


        att_idx, att_list_1, att_list_2 = self.make_att_dict(padded_sequences, nominal_heads)
        return (padded_sequences, 
            np.asarray(nominal_positions1), np.asarray(nominal_positions2), 
            att_idx, att_list_1, att_list_2,
            self.Y)


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
    def fit_to_window2(self, sequences, nominal_heads):
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
            #old_head1, old_head2 = nominal_heads[seq_idx]
            #nominal_heads[seq_idx] = old_head1 - h, old_head2 - h
        return pad_sequences(padded_sequence, self.clipping_value, value=0)


    def reverse_sequence(self, seqs):
        inv_map = self.reverse_word_idx()
        return [[inv_map[s] for s in xs] for xs in seqs]        

    def clean_markups(self,seqs):
        rev_seq = self.reverse_sequence(seqs)
        new_idx = self.word_idx()
        cleaned_seq = []
        for seq in rev_seq:
            cleaned_seq.append([new_idx[tok] for tok in seq if 
                tok != 'e1' and tok != 'e2'])
        return cleaned_seq

    '''
    makes the attention_one dictionary combination inputs
    '''
    def make_att_dict(self, seqs, nominals, fit=True):
        att_list_1 = []
        att_list_2 = []

        def add_to_dict_and_list(pair, att_list, fit=True):
            if fit and pair not in self.attentions_idx: 
                self.attentions_idx[pair] = len(self.attentions_idx) + 1 
            
            try:
                att_list.append(self.attentions_idx[pair])
            except KeyError:
                att_list.append(self.attentions_idx[(0,0)])
        ## Add OOV * OOV, or padding*padding for possible test combinations
        add_to_dict_and_list((0,0), [])


        for seq_idx, seq in enumerate(seqs):
            att_sub_list_1 = []
            att_sub_list_2 = []
            for tok_idx, tok in enumerate(seq):
                nominal_idx_1, nominal_idx_2 = nominals[seq_idx]
                pair_1 = min(tok, nominal_idx_1), max(tok, nominal_idx_1)
                pair_2 = min(tok, nominal_idx_2), max(tok, nominal_idx_2)

                add_to_dict_and_list(pair_1, att_sub_list_1, fit)
                add_to_dict_and_list(pair_2, att_sub_list_2, fit)
            att_list_1.append(att_sub_list_1)
            att_list_2.append(att_sub_list_2)

        return self.attentions_idx, np.asarray(att_list_1), np.asarray(att_list_2)

    def adjust_nominal_positions(self, X_pad, X_nom_heads):
        nominal_positions1 = []
        nominal_positions2 = []
        for seq_idx, seq in enumerate(X_pad):
            tmp_nom_pos1 = []
            tmp_nom_pos2 = []
            for token_idx, token in enumerate(seq):
                head1, head2 = X_nom_heads[seq_idx]
                nom_pos1 = (head1 - token_idx) + self.clipping_value - 1 
                nom_pos2 = (head2 - token_idx) + self.clipping_value - 1
                tmp_nom_pos1.append(nom_pos1)
                tmp_nom_pos2.append(nom_pos2)
            nominal_positions1.append(tmp_nom_pos1)
            nominal_positions2.append(tmp_nom_pos2)
        return np.array(nominal_positions1), np.array(nominal_positions2)


    ''' cut sentences above a maximum nominal head distance '''
    def clip_sentences(self, 
                       nominal_heads, 
                       sequences):

        logging.debug("Removing sentences outside max length..")

        self.idx_to_keep = []
        for seq_idx, seq in enumerate(nominal_heads):
            if (seq[1] - seq[0]) + 2 < self.clipping_value:
                self.idx_to_keep.append(seq_idx)                

        self.Y = np.asarray(self.Y)[self.idx_to_keep]

        return (np.asarray(nominal_heads)[self.idx_to_keep],
               np.asarray(sequences)[self.idx_to_keep])

# classes in the problem
output_dict = {
    "Cause-Effect(e1,e2)" : 0,
    "Cause-Effect(e2,e1)" : 1,
    "Instrument-Agency(e1,e2)" : 2,
    "Instrument-Agency(e2,e1)" : 3,
    "Product-Producer(e1,e2)" : 4,
    "Product-Producer(e2,e1)" : 5,
    "Content-Container(e1,e2)" : 6,
    "Content-Container(e2,e1)" : 7,
    "Entity-Origin(e1,e2)" : 8,
    "Entity-Origin(e2,e1)" : 9,
    "Entity-Destination(e1,e2)" : 10,
    "Entity-Destination(e2,e1)" : 11,
    "Component-Whole(e1,e2)" : 12,
    "Component-Whole(e2,e1)" : 13,
    "Member-Collection(e1,e2)" : 14,
    "Member-Collection(e2,e1)" : 15,
    "Message-Topic(e1,e2)" : 16,
    "Message-Topic(e2,e1)" : 17,
    "Other" : 18
}

reverse_dict = {v: k for k, v in output_dict.items()}


