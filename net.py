import torch
import numpy as np
import pandas as pd
import pickle
from examples_analysis import print_example
device = torch.device('cpu')
torch.manual_seed(256)
np.random.seed(256)

#---------------LOADING SPACY EMBEDDINGS FOR TEXTS-----
with open("test_embeddings_new.p", "rb") as f:
    data = pickle.load(f)
    test_embeddings_new = data

with open("embeddings_new.p", "rb") as f:
    data = pickle.load(f)
    embeddings_new = data

embeddings = torch.cat((embeddings_new, test_embeddings_new))
idx = int(embeddings.shape[0] * 0.05)
test_embeddings_new = embeddings[:idx]
embeddings_new = embeddings[idx:]


with open("targets_train.p", "rb") as f:
    data = pickle.load(f)
    targets_train = data

with open("targets_test.p", "rb") as f:
    data = pickle.load(f)
    targets_test = data

targets = torch.cat((targets_train, targets_test))
targets_test = targets[:idx]
targets_train = targets[idx:]

#test_data = pd.read_csv("test_data.csv")
valid_data = pd.read_csv("Validation.tsv", sep='\t')

with open("validation_embeddings.p", "rb") as f:
    data = pickle.load(f)
    valid_embeddings = data
valid_embeddings = [e.unsqueeze(0) for e in valid_embeddings]
valid_embeddings = torch.cat(valid_embeddings, dim=0)
print(valid_embeddings.shape)

class net(torch.nn.Module):
    def __init__(self, n_hidden_neurons, in_features, out_features):
        super(net, self).__init__()
        self.layer1 = torch.nn.Linear(in_features, n_hidden_neurons)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(n_hidden_neurons, int(n_hidden_neurons/1))
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(int(n_hidden_neurons/1), out_features)
        self.act_out = torch.nn.Sigmoid()
        self.lr = 1e-4 / 3
        self.optimizer = torch.optim.Adam(net.parameters(self), lr=self.lr)

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
def train(dili_net, X, Y, X_test, Y_test, batch_size=13, epochs=87):
    for epoch in range(epochs):
        order = np.random.permutation(len(X))
        for start_index in range(0, len(X), batch_size):
            dili_net.optimizer.zero_grad()
            batch_indices = order[start_index:start_index+batch_size]
            x_batch = X[batch_indices]
            y_batch = torch.LongTensor( Y[batch_indices] )
            preds = dili_net.forward(x_batch)
            loss_value = loss(preds.float(), y_batch.float())
            loss_value.backward()
            dili_net.optimizer.step()

        if epoch % 2 == 0:
            test_preds = dili_net.forward(X_test)
            test_preds = torch.where(test_preds > 0.5, 1, 0)
            print("epoch", epoch)
            accuracy = (test_preds == Y_test).float().mean().numpy()
            print("Accuracy:", accuracy)
            if accuracy >= 0.96:
                for param_group in dili_net.optimizer.param_groups:
                    param_group['lr'] = 1e-4/3
            a = input()
            if a == 's':
                break

    incorrect = (test_preds != Y_test)
    correct = (test_preds == Y_test)
    incorrect_indices = np.where(incorrect == True)
    correct_indices = np.where(correct == True)
    print("Incorrect:", len(incorrect_indices[0])/len(Y_test))
    print("Correct:", len(correct_indices[0])/len(Y_test))
    return incorrect_indices[0]

dili_net = net(300, 202, 1)
dili_net.to(device)

incorrect_indices = train(dili_net, X=embeddings_new, Y=targets_train, X_test=test_embeddings_new, Y_test=targets_test,
        batch_size=10, epochs=250)

results = dili_net.forward(valid_embeddings)
results = torch.where(results > 0.5, 1, 0)
print(torch.sum(results))
valid_data["Label"] = results.detach().numpy()
valid_data.to_csv("arsentii.ivasiuk@gmail.com_resultsNN.csv")
print(valid_data[:200])

