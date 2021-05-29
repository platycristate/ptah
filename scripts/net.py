import torch
import numpy as np
import pandas as pd
import pickle
from NN import train, net
from examples_analysis import print_example
device = torch.device('cuda:0')
torch.manual_seed(256)
np.random.seed(256)
path = "../data/"

#with open(path + "pmiidf_net.p") as f:
#    model = pickle.load(f)
#    pmiidf_net = model

#---------------LOADING SPACY EMBEDDINGS FOR TEXTS-----
with open(path +"large_train_embeddings.p", "rb") as f:
    data = pickle.load(f)
    embeddings_new = data

embeddings = torch.cat([e.unsqueeze(0) for e in embeddings_new], dim=0)

####################################################################
indices = np.random.permutation(embeddings.shape[0])
print(embeddings.shape)
idx = int(embeddings.shape[0] * 0.2)

embeddings = embeddings[indices]

test_embeddings_new = embeddings[-idx:].to(device)
embeddings_new = embeddings[:-idx].to(device)

with open(path + "targets_train.p", "rb") as f:
    data = pickle.load(f)
    targets_train = data

with open(path + "targets_test.p", "rb") as f:
    data = pickle.load(f)
    targets_test = data

targets = pd.read_csv(path + "DILI_data.csv")["Label"].values
targets = targets[indices]
targets = torch.from_numpy(targets).unsqueeze(1)

print(targets.shape)
targets_test = targets[-idx:].to(device)
targets_train = targets[:-idx].to(device)

#test_data = pd.read_csv("test_data.csv")
valid_data = pd.read_csv(path + "Validation.tsv", sep='\t')

with open(path + "large_validation_embeddings.p", "rb") as f:
    data = pickle.load(f)
    valid_embeddings = data

valid_embeddings = [e.unsqueeze(0) for e in valid_embeddings]
valid_embeddings = torch.cat(valid_embeddings, dim=0).to(device)
print(valid_embeddings.shape)

#####################################################
loss = torch.nn.BCELoss().to(device)


dili_net = net(400, 202, 1, lr=1e-4/3)
dili_net.to(device)

train(dili_net, X=embeddings_new.to(device), Y=targets_train.to(device),
        X_test=test_embeddings_new.to(device), Y_test=targets_test.to(device),
        loss=loss,
        batch_size=500, epochs=7000)

results = dili_net.forward(valid_embeddings)
results = torch.where(results > 0.5, 1, 0)
valid_data["Label"] = results.cpu().detach().numpy()
#valid_data.to_csv(path + "arsentii.ivasiuk@gmail.com_resultsNN.csv")
print(valid_data[:200])

