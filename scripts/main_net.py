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

#####################################################################
with open(path +"large_train_embeddings2.p", "rb") as f:
    data = pickle.load(f)
    embeddings_new = data

with open(path + "large_validation_embeddings2.p", "rb") as f:
    data = pickle.load(f)
    valid_embeddings = data

valid_data = pd.read_csv(path + "Validation.tsv", sep='\t')
targets = pd.read_csv(path + "DILI_data.csv")["Label"].values

######################################################################
# Embeddings for train data and test data
embeddings = torch.cat([e.unsqueeze(0) for e in embeddings_new], dim=0)
indices = np.random.permutation(embeddings.shape[0])
print(embeddings.shape)
idx = int(embeddings.shape[0] * 0.8)
embeddings = embeddings[indices]
embeddings_new = embeddings[:idx].to(device)
test_embeddings_new = embeddings[idx:].to(device)

embeddings_new = embeddings_new[:,:-2]
test_embeddings_new = test_embeddings_new[:,:-2]

# Targets 
targets = targets[indices]
targets = torch.from_numpy(targets).unsqueeze(1)
print(targets.shape)
targets_train = targets[:idx].to(device)
targets_test = targets[idx:].to(device)


# Embeddings for validation data
valid_embeddings = [e.unsqueeze(0) for e in valid_embeddings]
valid_embeddings = torch.cat(valid_embeddings, dim=0).to(device)
valid_embeddings = valid_embeddings[:, :-2]
print(valid_embeddings.shape)


#######################################################################
loss = torch.nn.BCELoss().to(device)
dili_net = net(400, 200, 1, lr=1e-4)
dili_net.to(device)

train(dili_net, X=embeddings_new.to(device), Y=targets_train.to(device),
        X_test=test_embeddings_new.to(device), Y_test=targets_test.to(device),
        loss=loss,
        batch_size=1000, epochs=7000)

results = dili_net.forward(valid_embeddings)
results = torch.where(results > 0.5, 1, 0)
valid_data["Label"] = results.cpu().detach().numpy()
#valid_data.to_csv(path + "arsentii.ivasiuk@gmail.com_resultsNN.csv")
print(valid_data[:200])

