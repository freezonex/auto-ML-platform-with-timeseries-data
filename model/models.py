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

class BaseRNNModel(Model):
    class RNNNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers, rnn_type):
            super().__init__()
            if rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            elif rnn_type == 'GRU':
                self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            else:
                raise ValueError("Unsupported RNN type")
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            rnn_out, _ = self.rnn(x)
            output = self.fc(rnn_out[:, -1, :])  # Get the last time step's output
            return output

    def __init__(self, input_dim, output_dim, rnn_type):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.model = None
        self.epochs = None
        self.batch_size = None
        self.optimizer = None
        self.criterion = nn.L1Loss()

    def set_params(self, hidden_dim=30, num_layers=1, lr=0.001, epochs=100, batch_size=32):
        self.model = self.RNNNet(self.input_dim, hidden_dim, self.output_dim, num_layers, self.rnn_type).cuda()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self, X, y):
        self.model.train()
        X = torch.tensor(X, dtype=torch.float32).cuda()
        y = torch.tensor(y, dtype=torch.float32).cuda()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = 0.5 * self.criterion(outputs, labels.unsqueeze(-1)) + 0.5 * nn.MSELoss()(outputs, labels.unsqueeze(-1))
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X).detach().cpu().numpy()
        return outputs

    def evaluate(self, X, y):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).cuda()
        y = torch.tensor(y, dtype=torch.float32).cuda()
        with torch.no_grad():
            outputs = self.model(X)
            loss = nn.MSELoss()(outputs, y).detach().cpu().numpy()
        return loss

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
        self.model.eval()


class LSTMModel(BaseRNNModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, 'LSTM')

class GRUModel(BaseRNNModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, 'GRU')
# param_grid = {
#     'LSTM':
#     {
#     'hidden_dim': [50, 25],
#     'num_layers': [1],
#     'lr': [0.001, 0.01],
#     'batch_size': [16, 32],
#     'epochs': [50, 100]}
# }
param_grid = {
    'LSTM':
    {
    'hidden_dim': [50],
    'num_layers': [1],
    'lr': [0.01],
    'batch_size': [ 32],
    'epochs': [20]},
    'GRU':
{
    'hidden_dim': [50],
    'num_layers': [1],
    'lr': [0.01],
    'batch_size': [ 32],
    'epochs': [20]},
}
