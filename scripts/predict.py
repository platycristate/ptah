import pandas as pd
import pmi_tfidf_classifier as ptic
import numpy as np
path = "../data/"
np.random.seed(250)

test_data = pd.read_csv(path + 'AdditionalDILItest.tsv', sep="\t")
train_data = pd.read_csv(path + "DILI_data.csv")
targets_train = train_data['Label'].values

tokenized_texts = ptic.tokenization(train_data)
tokenized_test_texts = ptic.tokenization(test_data)
N = len(tokenized_texts)

word2text_count = ptic.get_word_stat( tokenized_texts )
words_pmis = ptic.create_pmi_dict(tokenized_texts, targets_train, min_count=5)
results = ptic.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N)

test_data["Labels"] = results
test_data.to_csv(path + "arsentii.ivasiuk@gmail.com_additional.csv")




