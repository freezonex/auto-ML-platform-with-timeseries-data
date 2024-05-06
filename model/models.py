import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from abc import ABC, abstractmethod
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self,X,y):
        pass
    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class LSTMModel(Model):
    class LSTMNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return output

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.criterion = nn.MSELoss()
        self.model = None
        self.epochs = None
        self.batch_size = None
        self.optimizer = None

    def set_params(self,hidden_dim=30, num_layers=1, lr=0.001, epochs=100, batch_size=32):
        self.model = self.LSTMNet(self.input_dim, hidden_dim, self.output_dim, num_layers).cuda()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self, X, y):
        self.model.train()
        X = torch.tensor(X,dtype=torch.float32).cuda()
        y = torch.tensor(y,dtype=torch.float32).cuda()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            return outputs  # Direct output for regression

    def evaluate(self, X,y):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).cuda()
        y = torch.tensor(y, dtype=torch.float32).cuda()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
        return loss  # Direct output for regression

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
        self.model.eval()

param_grid = {
    'LSTM':
    {
    'hidden_dim': [50, 100, 150],
    'num_layers': [1, 2],
    'lr': [0.001, 0.01],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150]}
}
