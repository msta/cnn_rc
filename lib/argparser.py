import argparse

def build_argparser():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument("--debug", 
                        action="store_true")
    parser.add_argument("--no_pos",  
                        action="store_true",
                        default=False)
    parser.add_argument("-f", "--folds", 
                        type=int,
                        default=10)
    parser.add_argument("-d", "--dataset",
                        type=str,
                        choices=["semeval, ace2005"])
    parser.add_argument("-a1", "--attention_one", 
                        action="store_true",
                        default=False)
    parser.add_argument("-a2", "--attention_two", 
                        action="store_true",
                        default=False)
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10)
    parser.add_argument("--clipping",
                        type=int,
                        default=18)
    parser.add_argument("-r", "--rand",  
                        action="store_true")
    parser.add_argument("--markup",  
                        action="store_true")
    parser.add_argument("-o", "--optimizer",
                        type=str,
                        default='ada',
                        choices=["sgd", "ada"])
    parser.add_argument("--windowsize",
                        type=int,
                        default=400)
    parser.add_argument("--wordembeddingdim",
                        type=int,
                        default=300)
    parser.add_argument("--posembeddingdim",
                        type=int,
                        default=50)
    parser.add_argument("-loss",
                        type=str,
                        default="categorical_crossentropy")
    parser.add_argument("-dropoutrate",
                        type=float,
                        default=0.5)
    return parser
