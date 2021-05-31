'''
DATA PREPARATION
+1. Calculate pmi_dict for train data.
+2. Convert pmi_dict to the format word: [x, y] --> pmis_list = [(word, x, y) ....]
+3. Create embeddings for words in pmis_list --> embeddings = [[n1, n2, ..., nN], ....]
+4. Cut a piece of embeddings and pmis_list for testing (20%)

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
from NN import net2, train2
from examples_analysis import print_example
import pmi_tfidf_classifier as ptic
device = torch.device('cuda:0')
torch.random.seed()
np.random.seed(256)
path = "../data/"

#spacy.prefer_gpu()
spacy.require_gpu()
nlp = spacy.load("en_core_sci_lg", disable=['ner', 'parser'])

data = pd.read_csv(path + 'DILI_data_mixed.csv')
targets_data = data["Label"].values
tokenized_texts = ptic.tokenization(data)
idx = int(len(tokenized_texts) * 0.8)
tokenized_texts_train = tokenized_texts[:idx]

N = len(tokenized_texts_train)
words_pmis = ptic.create_pmi_dict(tokenized_texts_train, targets_data, min_count=1)
word2text_count = ptic.get_word_stat( tokenized_texts_train )

all_words = list(words_pmis[0].keys()) + list(words_pmis[1].keys())
words_both_classes = set(all_words)

#word2pmiidf = []
#
#for word in words_both_classes:
#    idf = np.log(N/word2text_count[word])
#    pmi0 = words_pmis[0][word]
#    pmi1 = words_pmis[1][word]
#    word2pmiidf.append((word, pmi0 * idf, pmi1 * idf))

print("Creating embeddings....")
#word_embeddings = [torch.as_tensor(nlp(i[0]).vector).unsqueeze(0)
#                    for i in word2pmiidf]

with open(path + "word_embeddings_scilg_cleaned.p", "rb") as f:
    data = pickle.load(f)
    word_embeddings = data

#word_embeddings = torch.cat(word_embeddings, dim=0)
print(word_embeddings.shape)
train_embeddings = word_embeddings[:idx]
test_embeddings = word_embeddings[idx:]

#pmiidf = torch.cat([torch.Tensor([i[1],i[2]]).unsqueeze(0)
#                    for i in word2pmiidf], dim=0)

with open(path + "pmiidf_scilg_cleaned.p", "rb") as f:
    data = pickle.load(f)
    pmiidf = data

print(pmiidf.shape)
train_pmiidf = pmiidf[:idx]
test_pmiidf = pmiidf[idx:]


loss = torch.nn.MSELoss().to(device)

dili_net = net2(50, 200, 2, lr=1e-4)
dili_net.to(device)

train2(dili_net, X=train_embeddings.to(device), Y=train_pmiidf.to(device),
        X_test=test_embeddings.to(device), Y_test=test_pmiidf.to(device),
        loss=loss,
        batch_size=1000, epochs=2000)

test_preds = dili_net.forward( test_embeddings.to(device) )
print(test_preds[25:30])
print(test_pmiidf[25:30])
torch.save(dili_net, path + "pmiidf_model.pt")


