import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
import pickle
import time
from collections import defaultdict
import pmi_tfidf_classifier as ptic
path = "../data/"

t1 = time.time()

pd.set_option("display.max_rows", None, "display.max_columns", None)
np.random.seed(250)

#spacy.prefer_gpu()
#nlp = spacy.load("en_core_sci_sm", disable=['ner', 'parser'])

data_raw = pd.read_csv(path + "DILI_data.csv")
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
words_pmis = ptic.create_pmi_dict(tokenized_texts, targets_train, min_count=5)
t2 = time.time()

t3 = time.time()
results = ptic.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N)
t4 = time.time()

# Assasin
precision = np.sum( np.logical_and(results, targets_test) ) / np.sum(results)

# Eisenhower
recall = np.sum( np.logical_and(results, targets_test) ) / np.sum(targets_test)

accuracy = (results == targets_test).mean()
FP_rate =  ((results - targets_test) == 1).sum()/np.sum(targets_test)
FN_rate = ((results - targets_test) == -1).sum()/np.sum(targets_test == 0)

F1 = 2 * (recall * precision)/(recall + precision)

print("Accuracy: %s\t \nPrecision: %s\t \nRecall: %s\t \nF1: %s\t"  % (accuracy, precision, recall, F1))
print("FP: %s\t \nFN: %s\t" % (FP_rate, FN_rate))
print("Training time: %s\t min \nClassification time: %s\t min" % (round((t2 - t1)/60, 3), round((t4 - t3)/60, 3)))

# Saving incorrectly classified examples
incorrect = np.where(results != targets_test)[0]
incorrect_examples = test_data.iloc[list(incorrect)]
incorrect_examples.to_csv(path + "incorrect.csv")




