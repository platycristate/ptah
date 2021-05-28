import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import spacy
import pickle
from collections import defaultdict
import pmi_tfidf_classifier as ptic
from tqdm import tqdm

filename = sys.argv[1]
output = sys.argv[2]

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_md", disable=['ner', 'parser'])
device = torch.device('cpu')

def text_embeddings(text_tokenized, words_pmis, word2text_count, N):
    embeddings = []
    for words in tqdm(text_tokenized):
        word2tfidf = ptic.get_doc_tfidf(words, word2text_count, N)
        embedding = torch.FloatTensor(np.zeros( nlp(text_tokenized[0][0]).vector.shape[0] + 2)).to(device)
        pmi0 = 0;
        pmi1 = 0;
        for word in words:
            embedding[:200] += torch.FloatTensor(nlp(word).vector).to(device)
            try:
                pmi0 += words_pmis[0][word] * word2tfidf[word]
                pmi1 += words_pmis[1][word] * word2tfidf[word]
            except:
                continue
        embedding[-1] = pmi0
        embedding[-2] = pmi1
        embeddings.append(embedding / len(words))
    return embeddings

# texts to be converted to embeddings
data = pd.read_csv(filename, sep="\t")
# train data needed to generate pmi dictionary
train_data = pd.read_csv('DILI_data.csv')
targets_train = train_data["Label"].values

# divide texts into tokens
tokenized_texts_train =  ptic.tokenization(train_data)
tokenized_texts = ptic.tokenization(data)

# calculates words frequency in texts
word2text_count = ptic.get_word_stat( tokenized_texts_train )
N = len(tokenized_texts_train)

# creates PMIs dictionary
words_pmis = ptic.create_pmi_dict(tokenized_texts_train,
        targets_train, min_count=5)

# calculate embeddings
embeddings = text_embeddings(tokenized_texts,
        words_pmis, word2text_count, N)

# save embeddings
with open(output, "wb") as file:
    pickle.dump(embeddings, file)
