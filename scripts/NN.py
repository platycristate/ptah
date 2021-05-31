import torch
import numpy as np
import pandas as pd

class net(torch.nn.Module):
    def __init__(self, n_hidden_neurons, in_features, out_features, lr):
        super(net, self).__init__()
        self.layer1 = torch.nn.Linear(in_features, n_hidden_neurons)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(n_hidden_neurons, int(n_hidden_neurons/3))
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(int(n_hidden_neurons/3), out_features)
        self.act_out = torch.nn.Sigmoid()
        self.lr = lr
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

class net2(torch.nn.Module):
    def __init__(self, n_hidden_neurons, in_features, out_features, lr):
        super(net2, self).__init__()
        self.layer1 = torch.nn.Linear(in_features, n_hidden_neurons)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(n_hidden_neurons, out_features)
        self.lr = lr
        self.optimizer = torch.optim.Adam(net2.parameters(self), lr=self.lr)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x

def train(dili_net, X, Y, X_test, Y_test, loss, batch_size=13, epochs=87):
    for epoch in range(epochs):
        order = np.random.permutation(len(X))
        for start_index in range(0, len(X), batch_size):
            dili_net.optimizer.zero_grad()
            batch_indices = order[start_index:start_index+batch_size]
            x_batch = X[batch_indices]
            #y_batch = torch.LongTensor( Y[batch_indices] ).to(device)
            y_batch =  Y[batch_indices]
            preds = dili_net.forward(x_batch)
            loss_value = loss(preds.float(), y_batch.float())
            loss_value.backward()
            dili_net.optimizer.step()

        if epoch % 10 == 0:
            test_preds = dili_net.forward(X_test)
            test_preds = torch.where(test_preds > 0.5, 1, 0)
            print("epoch", epoch)
            accuracy = (test_preds == Y_test).cpu().float().mean().numpy()
            print("Accuracy:", accuracy)
            if accuracy >= 0.95:
                print("Accuracy:", accuracy)
                a = input()
                if a == 's':
                    break
def train2(dili_net, X, Y, X_test, Y_test, loss, batch_size=13, epochs=87):
    for epoch in range(epochs):
        order = np.random.permutation(len(X))
        for start_index in range(0, len(X), batch_size):
            dili_net.optimizer.zero_grad()
            batch_indices = order[start_index:start_index+batch_size]
            x_batch = X[batch_indices]
            y_batch =  Y[batch_indices]
            preds = dili_net.forward(x_batch)
            loss_value = loss(preds.float(), y_batch.float())
            loss_value.backward()
            dili_net.optimizer.step()

        if epoch % 2 == 0:
            test_preds = dili_net.forward(X_test)
            print("epoch", epoch)
            loss_test = loss(test_preds, Y_test)
            print("Loss:", loss_test)
            a = input()
            if a == "s":
                break

