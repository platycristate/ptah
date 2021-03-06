import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import spacy
import pickle
from collections import defaultdict
from NN import net2
import torch
import pmi_tfidf_classifier as ptic
from tqdm import tqdm
path = "../data/"

filename = sys.argv[1]
output = sys.argv[2]
#spacy.require_gpu()
nlp = spacy.load("en_core_sci_lg", disable=['ner', 'parser', 'lemmatizer', 'attribute_ruler'])
device = torch.device('cuda:0')
# model that predicts PMI * IDF for words
# that are not in the PMI dictionary
model = torch.load(path + "pmiidf_model.pt")
model.to(device)

def text_embeddings(text_tokenized, words_pmis, word2text_count, N):
    embeddings = []
    for words in tqdm(text_tokenized):
        word2tfidf = ptic.get_doc_tfidf(words, word2text_count, N)
        embedding = torch.FloatTensor(np.zeros( nlp(text_tokenized[0][0]).vector.shape[0] + 2)).to(device)
        pmi0 = 0;
        pmi1 = 0;
        for word in words:
            tf = len(words)
            word_emb = torch.FloatTensor( nlp(word).vector )
            embedding[:200] += word_emb.to(device)
            if word in words_pmis[0]:
                pmi0 += words_pmis[0][word] * word2tfidf[word]
            if word in words_pmis[1]:
                pmi1 += words_pmis[1][word] * word2tfidf[word]
            else:
                predicted_pmiidf = model.forward( word_emb.to(device) ).cpu().detach().numpy()
                pmi0 += predicted_pmiidf[0] * tf
                pmi1 += predicted_pmiidf[1] * tf
        embedding[-1] = pmi0
        embedding[-2] = pmi1
        embeddings.append(embedding / len(words))
    return embeddings

# texts to be converted to embeddings
data = pd.read_csv(path + filename, sep=",")

# train data needed to generate pmi dictionary
train_data = pd.read_csv(path + 'DILI_data_mixed.csv')
targets_train = train_data["Label"].values

# divide texts into tokens
tokenized_texts_data =  ptic.tokenization(train_data)
idx = int(0.8 * len(tokenized_texts_data))

tokenized_texts_train = tokenized_texts_data[:idx]
tokenized_texts = ptic.tokenization(data)

# calculates words frequency in texts
word2text_count = ptic.get_word_stat( tokenized_texts_train )
N = len(tokenized_texts_train)

# creates PMIs dictionary
words_pmis = ptic.create_pmi_dict(tokenized_texts_train,
        targets_train, min_count=1)

# calculate embeddings
embeddings = text_embeddings(tokenized_texts, words_pmis, word2text_count, N)

# save embeddings
with open(path + output, "wb") as file:
    pickle.dump(embeddings, file)

