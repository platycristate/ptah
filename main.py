import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
import pickle
from collections import defaultdict
import pmi_tfidf_classifier as ptic
pd.set_option("display.max_rows", None, "display.max_columns", None)
np.random.seed(250)

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_sm", disable=['ner', 'parser'])

data_raw = pd.read_csv('DILI_data.csv')
indices = np.random.permutation(data_raw.index)
data = data_raw.loc[indices]
data = data_raw.sample(frac=1)
idx = int(data.shape[0] * 0.2)
test_data = data.iloc[:idx]
train_data = data.iloc[idx:]
targets_train = train_data['Label'].values
targets_test = test_data['Label'].values

tokenized_texts = ptic.tokenization(train_data)
tokenized_test_texts = ptic.tokenization(test_data)
N = len(tokenized_texts)
word2text_count = ptic.get_word_stat( tokenized_texts )
words_pmis = ptic.create_pmi_dict(tokenized_texts, targets_train, min_count=1)
results = ptic.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N)

precision = np.sum( np.logical_and(results, targets_test) ) / np.sum(targets_test)
accuracy = (results == targets_test).mean()
print("Accuracy: %s \nPrecision: %s" % (accuracy, precision))

words_pmis_df = pd.DataFrame.from_dict(words_pmis)
print(words_pmis_df[:200])




