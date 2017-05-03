import argparse

def build_argparser():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument("--debug", 
                        action="store_true")
    parser.add_argument("--wordnet",
                        action="store_true",
                        default=False)
    parser.add_argument("--embedding",
                        type=str,
                        choices=["word2vec", "glove", "rand"],
                        default="word2vec"),
    parser.add_argument("--merge",
                        type=bool,
                        default=False)
    parser.add_argument("--no_pos",  
                        action="store_true",
                        default=False)
    parser.add_argument("--exclude_other",  
                        action="store_true",
                        default=False)
    parser.add_argument("-t", "--train_file",
                        type=str,
                        default="trainfile_clean_seg_lower")
    parser.add_argument("--test_file",
                        type=str,
                        default="test_clean_seg_lower")
    parser.add_argument("-a", "--aug_file",
                        type=str)
    parser.add_argument("-f", "--folds", 
                        type=int,
                        default=5)
    parser.add_argument("-wh", "--window_sizes",
                        type=int,
                        nargs="+",
                        default=[2,3,4,5])
    parser.add_argument("-d", "--dataset",
                        type=str,
                        choices=["semeval", "ace2005"],
                        default="semeval")
    parser.add_argument("-l2",
                        type=float,
                        default=0.0)
    parser.add_argument("-a1", "--attention_one", 
                        action="store_true",
                        default=False)
    parser.add_argument("-a2", "--attention_two", 
                        action="store_true",
                        default=False)
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=25)
    parser.add_argument("--clipping",
                        type=int,
                        default=15)
    parser.add_argument("--markup",  
                        action="store_true")
    parser.add_argument("-o", "--optimizer",
                        type=str,
                        default='ada',
                        choices=["sgd", "ada"])
    parser.add_argument("--filter_size",
                        type=int,
                        default=150)
    parser.add_argument("--wordembeddingdim",
                        type=int,
                        default=300)
    parser.add_argument("--posembeddingdim",
                        type=int,
                        default=50)
    parser.add_argument("--loss",
                        type=str,
                        default="categorical_crossentropy")
    parser.add_argument("--dropoutrate",
                        type=float,
                        default=0.5)
    return parser
