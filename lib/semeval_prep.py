import numpy as np
import math
import re

import logging

from .functions import debug_print, debug_print_dict
from .tokenizer import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Preprocessor():

    def __init__(self, 
        clipping_value=18,
        markup=False):

        
        self.n = clipping_value
        self.tokenizer = None
        self.texts = None
        self.nom1_idx = 1
        self.nom2_idx = 1
        self.attentions_idx = {}
        self.markup = markup
        self.Y = []
        self.n_values = []
        self.clipping_value = clipping_value
        self.oov_val = -1

    '''
    oov_val shows the number of words in the tokenizer
    with the first fitting
    '''
    def fit_tokenizer(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.texts)

        self.nom1_idx = self.tokenizer.word_index['e1']
        self.nom2_idx = self.tokenizer.word_index['e2']
        self.oov_val = len(self.tokenizer.word_index)
        

    
    def load_dataset(self, debug=False):

        if debug:
            return open("data/semeval/training/TRAIN_FILE_SMALL.txt", "r")
        else:
            return open("data/semeval/training/TRAIN_FILE.txt", "r") 

    def get_text(self, line):
        # split on quotes and get the raw text
        stripped = re.sub("\d{0,5}", "", line, count=1).strip()
        return stripped[1:len(stripped)-2]

    def get_id(self, line):
        return re.findall("\d{0,5}", line)[0]


    def read_dataset(self, debug=False):

        dataset = self.load_dataset(debug)

        X_raw = []
        Y = []
        i = 0
        for line in dataset:
            if i == 0:
                X_raw.append(self.get_text(line))
            if i == 1:
                Y.append(output_dict[line.strip()])
            if i == 2 or i == 3:
                pass
            i += 1
            if i % 4 == 0:
                i = 0
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
            data.append(get_text(line))
            ids.append(get_id(line))    
        return data, ids

    def glue_entities(self,texts):
        def add_underscore(ent):
            e1 = ""
            for e in ent.split(" "):
                e1 += e + "_"
            return e1[0:-1]
            
        def replace_ents(text):
            e1_org = text.split("<e1>")[1].split("</e1>")[0]
            e1_new = add_underscore(e1_org)
            e2_org = text.split("<e2>")[1].split("</e2>")[0]
            e2_new = add_underscore(e2_org)
            rep1 = text.replace(e1_org, e1_new)
            return rep1.replace(e2_org, e2_new)

        return [replace_ents(t) for t in texts]
        

                            
    
    def transform(self, texts, labels):

        self.Y = np.asarray(labels)

        texts = self.glue_entities(texts)

        sequences = self.tokenizer.texts_to_sequences(texts, 0)
        #Calculate nominal heads and tails 
        nominal_relations, nominal_heads, sequences = self.nominal_positions(sequences)
        #Remove sentences outside clipping value
        nominal_relations, nominal_heads, sequences_clip = self.clip_sentences(nominal_relations,
                                                                               nominal_heads,
                                                                               sequences)

        logging.debug("Maximum nominal distance: " + str(self.n))
        
        ## Adjust and remove markup
        if not self.markup:
            ## shift nominal relations to the left due to markups being removed
            nominal_relations = [(x[0]-1, x[1]-3) for x in nominal_relations]
            nominal_heads = [(x[0]-1, x[1]-3) for x in nominal_heads]
            sequences_clip = self.clean_markups(sequences_clip)


        padded_sequences = self.pad_and_heads(sequences_clip, nominal_relations, nominal_heads, self.n)
        nominal_positions1, nominal_positions2 = self.adjust_nominal_positions(padded_sequences, nominal_heads)         
        
        logging.debug("Attention dictionarys created with nominal HEADS only")

        att_idx, att_list_1, att_list_2 = self.make_att_dict(padded_sequences, nominal_heads, fit=False)
        
        debug_print(att_idx, "Attention Indices")
        debug_print(att_list_1, "Attention pair list 1")
        debug_print(att_list_2, "Attention pair list 2")

        
        return (padded_sequences, 
            nominal_positions1, nominal_positions2, 
             att_list_1, att_list_2, self.Y)
            


    def fit_transform(self, texts, labels):

        self.texts = texts
        self.fit_tokenizer()
        self.Y = np.asarray(labels)
        sequences = self.tokenizer.texts_to_sequences(texts)

        nominal_relations, nominal_heads, sequences = self.nominal_positions(sequences)
        nominal_relations, nominal_heads, sequences_clip = self.clip_sentences(nominal_relations,
                                                                               nominal_heads,
                                                                               sequences)

        logging.debug("Maximum nominal distance: " + str(self.n))

        if not self.markup:
            ## shift nominal relations to the left due to markups being removed
            nominal_relations = [(x[0]-1, x[1]-3) for x in nominal_relations]
            nominal_heads = [(x[0]-1, x[1]-3) for x in nominal_heads]
            sequences_clip = self.clean_markups(sequences_clip)

        padded_sequences = self.pad_and_heads(sequences_clip, nominal_relations, nominal_heads, self.n)
        nominal_positions1, nominal_positions2 = self.adjust_nominal_positions(padded_sequences, nominal_heads)         
        
        logging.debug("Attention dictionarys created with nominal HEADS only")

        att_idx, att_list_1, att_list_2 = self.make_att_dict(padded_sequences, nominal_heads)
        
        debug_print(att_idx, "Attention Indices")
        debug_print(att_list_1, "Attention pair list 1")
        debug_print(att_list_2, "Attention pair list 2")

        
        return (padded_sequences, 
            nominal_positions1, nominal_positions2, 
            att_idx, att_list_1, att_list_2,
            self.Y)

    ''' cut sentences above a maximum nominal relation distance '''
    def clip_sentences(self, 
                       nominal_relations, 
                       nominal_heads, 
                       sequences):

        logging.debug("Removing sentences outside max length..")

        self.idx_to_keep = []
        for seq_idx, seq in enumerate(nominal_relations):
            if seq[1] - seq[0] < self.clipping_value:
                self.idx_to_keep.append(seq_idx)                

        self.Y = self.Y[self.idx_to_keep]

        return (nominal_relations[self.idx_to_keep], 
               nominal_heads[self.idx_to_keep],
               sequences[self.idx_to_keep])

    ''' calculate nominal heads and tails '''
    def nominal_positions(self, sequences):
        nominal_heads = []
        nominal_relations = []
        self.idx_to_keep = []
        for seq_idx, seq in enumerate(sequences):
            nom1_head = -1
            nom2_head = -1
            nom2_tail = -1
            for token_idx, token in enumerate(seq):
                if token == self.nom1_idx and nom1_head == -1:
                    nom1_head = token_idx + 1
                if token == self.nom2_idx:
                    if nom2_head == -1:
                        nom2_head = token_idx + 1
                    else:
                        nom2_tail = token_idx - 1
            nominal_relations.append((nom1_head, nom2_tail))
            nominal_heads.append([nom1_head, nom2_head])
        
        return (np.asarray(nominal_relations),
               np.asarray(nominal_heads),
               np.asarray(sequences))
        

    def word_idx(self):
        if self.markup:
            return self.tokenizer.word_index
        else:
            return {k: v-2 for k, v in self.tokenizer.word_index.items()
            if v != self.nom1_idx and v != self.nom2_idx  } 


    def pad_and_heads(self, sequences, nominal_relations, nominal_heads, n):
        padded_sequence = []
        for seq_idx, seq in enumerate(sequences):
            head, tail = nominal_relations[seq_idx]
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

            padded_sequence.append( seq[h:t])
            old_head1, old_head2 = nominal_heads[seq_idx]
            nominal_heads[seq_idx] = old_head1 - h, old_head2 - h
        return pad_sequences(padded_sequence, maxlen=n)

    def adjust_nominal_positions(self, X_pad, X_nom_heads):
        nominal_positions1 = []
        nominal_positions2 = []
        for seq_idx, seq in enumerate(X_pad):
            tmp_nom_pos1 = []
            tmp_nom_pos2 = []
            for token_idx, token in enumerate(seq):
                head1, head2 = X_nom_heads[seq_idx]
                nom_pos1 = (head1 - token_idx) + self.n - 1 
                nom_pos2 = (head2 - token_idx) + self.n - 1
                tmp_nom_pos1.append(nom_pos1)
                tmp_nom_pos2.append(nom_pos2)
            nominal_positions1.append(tmp_nom_pos1)
            nominal_positions2.append(tmp_nom_pos2)
        return np.array(nominal_positions1), np.array(nominal_positions2)

    def reverse_sequence(self, seqs):
        inv_map = {v: k for k, v in self.tokenizer.word_index.items()}
        return [[inv_map[s] for s in xs if s != 0] for xs in seqs]        

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





'''
    def find_n(self, nominal_relations):
        if not True: 
            n_values = [ x[1] - x[0] for x in nominal_relations]
            if not self.markup:
                n_values = [n-2 for n in n_values]
            self.n = max(n_values)
        else:
            self.n = self.clipping_value
'''

