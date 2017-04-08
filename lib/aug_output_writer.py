import numpy as np
from lib.semeval.classes import reverse_dict

kept_ids = set(kept_ids)
best_clz = [np.argmax(p) for p in preds]
best_prob = [np.max(p) for p in preds]



with open("data/semeval/training/web_examples_done99.txt", "w+") as f:
    correct_preds = 0
    pred_id = 0
    for kept_id, data_id in enumerate(kept_ids):
        sentence = data[data_id]
        if kept_id not in fails:
            org_clz = clzz[data_id]
            new_clz = best_clz[pred_id]
            if new_clz in org_clz and best_prob[pred_id] > 0.95:
                f.write(str(kept_id) + '\t"' + sentence + '"' + "\n")
                f.write(reverse_dict[new_clz] + "\n") 
                f.write("Comment: \n")
                f.write("\n")
                #f.write("Accuracy: " + str(best_prob[pred_id]) + "\n")
                correct_preds += 1
            pred_id += 1


print(correct_preds)
