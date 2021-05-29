'''
DATA PREPARATION
+1. Calculate pmi_dict for train data.
+2. Convert pmi_dict to the format word: [x, y] --> pmis_list = [(word, x, y) ....]
+3. Create embeddings for words in pmis_list --> embeddings = [[n1, n2, ..., nN], ....]
4. Cut a piece of embeddings and pmis_list for testing (20%)

NEURAL NETWORK
1. INPUT: word embedding (200, 1)
2. HIDDEN NEURONS: 50 neurons
3. OUTPUT: 1 pmi* idf (zero class), 2 pmi * idf (first class)
4. Activatin functions: Tanh
5. Loss Function: MSE and BCELoss

TRAIN AND TEST
'''
import torch
import numpy as np
import pandas as pd
import pickle
import spacy
from examples_analysis import print_example
import pmi_tfidf_classifier as ptic
device = torch.device('cpu')
torch.random.seed()
np.random.seed(256)

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_md", disable=['ner', 'parser'])

data_raw = pd.read_csv('DILI_data.csv')
data = data_raw.sample(frac=1)
targets_data = data["Label"].values
tokenized_texts = ptic.tokenization(data)

N = len(tokenized_texts)
words_pmis = ptic.create_pmi_dict(tokenized_texts, targets_data, min_count=5)
word2text_count = ptic.get_word_stat( tokenized_texts )

all_words = list(words_pmis[0].keys()) + list(words_pmis[1].keys())
words_both_classes = set(all_words)

word2pmiidf = []

for word in words_both_classes:
    idf = np.log(N/word2text_count[word])
    pmi0 = words_pmis[0][word]
    pmi1 = words_pmis[1][word]
    word2pmiidf.append((word, pmi0 * idf, pmi1 * idf))

word_embeddings = [torch.from_numpy(nlp(i[0]).vector).unsqueeze(0)
                    for i in word2pmiidf]
word_embeddings = torch.cat(word_embeddings, dim=0)
print(word_embeddings.shape)
idx = int( word_embeddings.shape[0] * 0.2 )
test_embeddings = word_embeddings[:idx]
train_embeddings = word_embeddings[idx:]

pmiidf = torch.cat([torch.Tensor([i[1],i[2]]).unsqueeze(0)
                    for i in word2pmiidf], dim=0)
print(pmiidf.shape)
test_pmiidf = pmiidf[:idx]
train_pmiidf = pmiidf[idx:]







