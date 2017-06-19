# cnn_rc
Relation Classification with Convolutional Neural Network on SemEval Task 8 2010

# Requirements

`` pip install -r requirements.txt ``

Some dependencies need to be installed manually:

The word_embeddings must be downloaded and put into the word_embeddings folder for the default experiment to work.
They should have the name ''word_embeddings.bin'' 
 
Tensorflow

# Usage
Example experiment from the root folder:
`` python -m lib.main --embedding word2vec --filter-size 250 ``
 
There are many different params which can be set. Check argparser.py or write me a mail.
Everyone who wants to run this experiment will be able to do so, just let me know if you are interested.
