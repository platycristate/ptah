import torch
import numpy as np
import pandas as pd
import pickle
from examples_analysis import print_example
device = torch.device('cpu')
torch.random.seed()
np.random.seed(256)

#---------------LOADING SPACY EMBEDDINGS FOR TEXTS-----
with open("test_embeddings_new.p", "rb") as f:
    data = pickle.load(f)
    test_embeddings_new = data

with open("embeddings_new.p", "rb") as f:
    data = pickle.load(f)
    embeddings_new = data

with open("targets_train.p", "rb") as f:
    data = pickle.load(f)
    targets_train = data

with open("targets_test.p", "rb") as f:
    data = pickle.load(f)
    targets_test = data

test_data = pd.read_csv("test_data.csv")

class net(torch.nn.Module):
    def __init__(self, n_hidden_neurons, in_features, out_features):
        super(net, self).__init__()
        self.layer1 = torch.nn.Linear(in_features, n_hidden_neurons)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(n_hidden_neurons, int(n_hidden_neurons/2))
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(int(n_hidden_neurons/2), out_features)
        self.act_out = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.layer1(x)
        #x = self.dropout(x)

        x = self.act1(x)
        x = self.layer2(x)

        x = self.act2(x)

        x = self.layer3(x)
        x = self.act_out(x)
        return x

loss = torch.nn.BCELoss().to(device)

def train(X, Y, X_test, Y_test, batch_size=13, epochs=87):
    for epoch in range(epochs):
        order = np.random.permutation(len(X))
        for start_index in range(0, len(X), batch_size):
            optimizer.zero_grad()
            batch_indices = order[start_index:start_index+batch_size]
            x_batch = X[batch_indices]
            y_batch = torch.LongTensor( Y[batch_indices] )
            preds = dili_net.forward(x_batch)
            loss_value = loss(preds.float(), y_batch.float())
            loss_value.backward()
            optimizer.step()

        if epoch % 50 == 0:
            test_preds = dili_net.forward(X_test)
            test_preds = torch.where(test_preds > 0.5, 1, 0)
            print("epoch", epoch)
            print("Accuracy:", (test_preds == Y_test).float().mean().numpy())

    incorrect = (test_preds != Y_test)
    correct = (test_preds == Y_test)
    incorrect_indices = np.where(incorrect == True)
    correct_indices = np.where(correct == True)
    print("Incorrect:", len(incorrect_indices[0])/len(Y_test))
    print("Correct:", len(correct_indices[0])/len(Y_test))
    return incorrect_indices[0]

dili_net = net(300, 202, 1)
dili_net.to(device)
optimizer = torch.optim.Adam(dili_net.parameters(),betas=(0.8, 0.999), lr=1e-4/3)
incorrect_indices = train(X=embeddings_new, Y=targets_train, X_test=test_embeddings_new, Y_test=targets_test,
        batch_size=1500, epochs=7500)

print_example(test_data, list(incorrect_indices))
