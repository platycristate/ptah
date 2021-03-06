import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
from time import time
import pickle
from collections import defaultdict
import pmi_tfidf_classifier as ptic
path = "../data/"

pd.set_option("display.max_rows", None, "display.max_columns", None)
np.random.seed(250)

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_sm", disable=['ner', 'parser'])

train_data = pd.read_csv(path + 'DILI_data_mixed.csv')
test_data = pd.read_csv(path + "Validation.tsv", sep="\t")

targets_train = train_data['Label'].values

tokenized_texts = ptic.tokenization(train_data)
tokenized_test_texts = ptic.tokenization(test_data)

N = len(tokenized_texts)
word2text_count = ptic.get_word_stat( tokenized_texts )
words_pmis = ptic.create_pmi_dict(tokenized_texts, targets_train, min_count=1)

t1 = time()
results = ptic.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N)
t2 = time()

test_data["Label"] = results
print("Classfication time: %s min" % (round(t2 - t1, 3)/60))

test_data.to_csv(path + "arsentii.ivasiuk@gmail.com_results.csv")





