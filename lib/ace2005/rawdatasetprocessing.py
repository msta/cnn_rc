
def go():
    positive_json = []
    for dirpath, dirname, filenames in os.walk("ace2005/training/cnn_json"):
        for filename in filenames:
            path, ext = os.path.splitext(filename)
            if ext == '.json':
                full_name = os.path.join(dirpath, filename)
                positive_json.append(full_name)

    word_dict = {}
    sentences = []
    entities = []
    labels = []
    positive_labels(positive_json, word_dict, entities, sentences, labels)
    negative_labels(negative_json, word_dict, entities, sentences, labels)
    write_csv(word_dict, sentences, entities, labels)

def go2():
    json_files = []
    for dirpath, dirname, filenames in os.walk("ace2005/training/json_full"):
        for filename in filenames:
            path, ext = os.path.splitext(filename)
            if ext == '.json':
                full_name = os.path.join(dirpath, filename)
                json_files.append(full_name)
    word_dict = {}
    sentences = []
    entities = []
    labels = []
    all_labels(json_files, word_dict, entities, sentences, labels)
    write_csv2(word_dict, sentences, entities, labels)

def write_csv2(word_dict, sentences, entities, labels):
    with open("ace2005/training/ace2.csv", "w+") as f:
        for idx, sent in enumerate(sentences):
            for tok_idx, tok in enumerate(sent):
                printstr = str(tok)
                if tok_idx < len(sent) - 1:
                    printstr += " "
                f.write(printstr)
            f.write("\t")
            e1, e2 = entities[idx]
            f.write(str(e1) + "\t" + str(e2) + "\t")
            label = labels[idx]
            f.write(str(label) + "\n")

    pickle.dump(word_dict, open("ace2005/training/vocab.pkl", "w+"))




def negative_labels(json_str, word_dict, head_list, sent_list, labels):
    data = json.loads(json_str, cls=ConcatJSONDecoder)
    for d in data:
        if d["relLabels"][0] != "NO_RELATION":
            continue
        sent = [get_token_id(word_dict, word) for word in d["words"]]
        e1 = d["nePairs"]["m1"]["head"]
        e2 = d["nePairs"]["m2"]["head"]
        head_list.append((e1,e2))
        sent_list.append(sent)
        labels.append("OTHER")

def all_labels(json_strs, word_dict, head_list, sent_list, labels):
    for json_str in json_strs:
        data = json.load(open(json_str), cls=ConcatJSONDecoder)
        for datapoint in data:
            sent = [get_token_id(word_dict, word) for word in datapoint["words"]]
            nepairs = json.loads(datapoint["nePairs"])[0]

            e1 = nepairs["m1"]["head"]
            e2 = nepairs["m2"]["head"]
            head_list.append((e1,e2))
            sent_list.append(sent)
            labels.append(datapoint["relLabels"][0])
    

# ###
class Entity():
    def __init__(self, uid, head, tok_list):
        self.uid = uid
        self.head = head
        self.tok_list = tok_list

def get_token_id(word_dict, word):
    word = word.lower()
    try:
        return word_dict[word]
    except KeyError:
        word_dict[word] = len(word_dict) + 1
        return word_dict[word]

def write_csv(word_dict, sentences, entities, labels):
    with open("ace2005/training/ace2.csv", "w+") as f:
        for idx, sent in enumerate(sentences):
            for tok_idx, tok in enumerate(sent):
                printstr = str(tok)
                if tok_idx < len(sent) - 1:
                    printstr += " "
                f.write(printstr)
            f.write("\t")
            e1, e2 = entities[idx]
            f.write(str(e1.head) + "\t" + str(e2.head) + "\t")
            label = labels[idx]
            f.write(str(label) + "\n")

    pickle.dump(word_dict, open("ace2005/training/vocab.pkl", "w+"))


def count_relations(domain_dir):
    json_list = []
    duplicates = set()
    for dirpath, dirname, filenames in os.walk(domain_dir):
        for filename in filenames:
            path, ext = os.path.splitext(filename)

            if filename.endswith(".apf.xml") and filename not in duplicates:
                duplicates.add(filename)
                full_name = os.path.join(dirpath, filename)
                json_list.append(full_name)
            elif filename in duplicates and len(duplicates) % 50 == 0:
                print "Duplicate grown by 50..." 

    counter = 0
    for k in json_list:
        stt = open(k).read()
        b = BeautifulSoup(stt, "xml")
        counter += len(b.findAll("relation"))

    return counter


def positive_labels(files):   
    
    entity_sent_list = []
    tok_sent_list = []
    label_list   = []
    word_dict = {}
    
    for path in files:

        file = json.load(open(path))

        raw_text = file["text"]

        mention_list = file["situationMentionSetList"][0]["mentionList"]
        entity_mentions = file["entityMentionSetList"][0]["mentionList"]

        for ment_dict in mention_list:
            try: 
                id_1, id_2, clz, token_id = process_mention(ment_dict)

                e1 = find_entity(id_1, entity_mentions)
                e2 = find_entity(id_2, entity_mentions)
                entity_sent_list.append((e1,e2))

                sentence = find_sentence(file["sectionList"], token_id)
                sent_start = sentence["textSpan"]["start"]
                sent_end = sentence["textSpan"]["ending"]
                sent_text =  raw_text[sent_start:sent_end+1]

                tok_sent = build_token_sentence(sentence, word_dict)
                tok_sent_list.append(tok_sent)
                label_list.append(clz)
            except:
                print "Broken Json", ment_dict["uuid"]
                continue

            
    
        
    return tok_sent_list ,  word_dict


''' adds words to the word_dict and sequences the words into integers '''
def build_token_sentence(sentence, word_dict):

    tok_list = []
    for tt in sentence["tokenization"]["tokenList"]["tokenList"]:
        tok = tt["text"]
        tok_id = get_token_id(word_dict, tok)
        tok_list.append(tok_id)
        

    return tok_list     


''' find a specific sentence based on its tokenization id '''
def find_sentence(section_list, token_id):
    for sec in section_list:
        for s in sec["sentenceList"]:
            if s["tokenization"]["uuid"]["uuidString"] == token_id:
                return s

    raise ValueError

''' finds entity by uuid from a JSON entity list '''
def find_entity(eid, entity_mentions):
    for f in entity_mentions:
        uid = f["uuid"]["uuidString"]
        if uid == eid:
            head = f["tokens"]["anchorTokenIndex"]
            tok_list = f["tokens"]["tokenIndexList"]
            return Entity(uid, head, tok_list)
    raise KeyError

def process_mention(ment_dict):

    entities = ment_dict["argumentList"]

    clz = ment_dict["situationKind"].split(":")[0]

    id_1 = entities[0]["entityMentionId"]["uuidString"]
    id_2 = entities[1]["entityMentionId"]["uuidString"]

    token_id = ment_dict["tokens"]["tokenizationId"]["uuidString"]
    return id_1, id_2, clz, token_id




#shameless copy paste from json/decoder.py
FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)
class ConcatJSONDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        s_len = len(s)

        objs = []
        end = 0
        while end != s_len:
            obj, end = self.raw_decode(s, idx=_w(s, end).end())
            end = _w(s, end).end()
            objs.append(obj)
        return objs
