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

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_lg", disable=['ner', 'parser'])
device = torch.device('cpu')

# model that predicts PMI * IDF for words
# that are not in the PMI dictionary
model = torch.load(path + "pmiidf_model.pt")

def text_embeddings(text_tokenized, words_pmis, word2text_count, N):
    embeddings = []
    for words in tqdm(text_tokenized):
        word2tfidf = ptic.get_doc_tfidf(words, word2text_count, N)
        embedding = torch.FloatTensor(np.zeros( nlp(text_tokenized[0][0]).vector.shape[0] + 2)).to(device)
        embedding = embedding.unsqueeze(0)
        pmi0 = 0;
        pmi1 = 0;
        for word in words:
            word_emb = torch.FloatTensor( nlp(word).vector ).unsqueeze(0)
            embedding[:200] += word_emb.to(device)
            try:
                pmi0 += words_pmis[0][word] * word2tfidf[word]
                pmi1 += words_pmis[1][word] * word2tfidf[word]
            except:
                predicted_pmis = model.forward( word_emb ).detach().numpy().squeeze(0)
                print(predicted_pmis)
                pmi0 += predicted_pmis[0]
                pmi1 += predicted_pmis[1]
        embedding[-1] = pmi0
        embedding[-2] = pmi1
        embeddings.append(embedding / len(words))
    return embeddings

# texts to be converted to embeddings
data = pd.read_csv(path + filename, sep="\t")

# train data needed to generate pmi dictionary
train_data = pd.read_csv(path + 'DILI_data.csv')
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

