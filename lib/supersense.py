
class SupersenseLookup():

    def __init__(self):
        self.sent_dict = {}
        self.supersense_id = {}


    def fit(self, fulltext, token_split=" ", sent_split="\n"):
    ''' must fit the .tags format so words are on a straight sentence
    with word|pos-tag|basic-form|supersense etc. fulltext is a full string with every sentence
    split by \n '''
        count = len(self.sent_dict)
        for t in fulltext.split(sent_split):
            lookup = {}
            token_count = 0
            current_word = None
            for token in t.split(token_split):
                if token_count == 0:
                    current_word = token
                if token_count == 3:
                    if token not in self.supersense_id:
                        self.supersense_id[token] = len(self.supersense_id) + 1
                    ss = self.supersense_id[token]
                    lookup[current_word] = ss 
                    token_count = 0
                else:
                    token_count += 1
            self.sent_dict[count] = lookup
            count += 1


    def transform(self, tokenized_sentences, begin=0):
        all_supersenses = []
        counter = begin
        for sent_idx, sent in enumerate(tokenized_sent):
            sent_supersense = []
            sent_lookup = self.sent_dict[sent_idx]
            for token in sent:
                try:
                    ss.append(sent_lookup[token])
                except KeyError:
                    ss.append(0)
            all_supersenses.append(sent_supersense)
        return all_supersenses