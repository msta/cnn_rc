    import numpy as np
    import pickle
    import ast
    import keras
    from lib.semeval.classes import *
    import re

    ## Used a punkt tokenizer for sentence splitting. Then simply counted tokens
    ## between entity searches. 

    clip_value =  16
    entity_pairs = ast.literal_eval(pickle.load(open('entity_pairs.txt', 'rb')))
    prepper = pickle.load(open('prepper.pkl', 'rb'))


    extra_data = open("extra_text01.txt").read().split("\n")

    model = keras.models.load_model("newest.model")


    sents = []
    sent_pairs = []

    pair_count = 0

    def get_reg(entity):
        return r'(?<![\w\-\/\–])'+entity+r'(?![\w\-\/\–])'

    for sent_idx , sent in enumerate(extra_data):
        if sent_idx % 500 == 0:
            print ("sent_idx" , sent_idx)
        for idx, (e1,e2) in enumerate(entity_pairs):
            if e1 in sent and e2 in sent:
                #negative lookahead for hyphens
                reg_e1 = get_reg(e1)
                reg_e2 = get_reg(e2)
                
                e1_match = re.search(reg_e1, sent)
                if e1_match:
                    try:
                        e1_start = e1_match.span()[0]
                        e1_end = e1_match.span()[1]
                        e2_match = re.search(reg_e2, sent[e1_end:])
                        if e2_match:
                            sent_end = e1_end + e2_match.span()[1]+1
                            sent_split = sent[e1_start:sent_end].split(" ")
                            if len(sent_split) > clip_value:
                                continue                     
                            pair_count += 1
                            if pair_count % 200 == 0:
                                print("pair_count" , pair_count)
                            sents.append(sent_idx)
                            sent_pairs.append(idx)
                    except:
                        print("something is wrong")

    print(pair_count)
    sent_with_ents = []

    for i, s in enumerate(sents):
        e1, e2 = entity_pairs[sent_pairs[i]]
        sent = extra_data[s]
        reg_e1 = get_reg(e1)
        reg_e2 = get_reg(e2)
        new_sent = re.sub(reg_e1,"<e1>" + e1 + "</e1><SPLIT>", sent, count=1)
        sub_sents = new_sent.split("<SPLIT>")
        new_sent = sub_sents[0] + re.sub(reg_e2, "<e2>" + e2 + "</e2>", sub_sents[1], count= 1)
        sent_with_ents.append(new_sent)

    seq, nom1, nom2, _, _, _, _  = prepper.transform(sent_with_ents, [])

    fails = []
    pred = model.predict([seq,nom1,nom2])

    # for i in range(0, len(seq)):
    #     try:
    #         pred = model.predict([seq[i:i+1],nom1[i:i+1],nom2[i:i+1]])
    #     except:
    #         fails.append(i)


    clz_int = [np.argmax(p) for p in pred]

    clz_str = [reverse_dict[c] for c in clz_int]

    values = [np.max(p) for p in pred]