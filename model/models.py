import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
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

    def predict(self, X, batch_size=32):
        self.model.eval()
        predictions = []

        # Avoid converting the entire array to a GPU tensor at once
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # Convert slices of arrays to tensors directly and move to GPU
                X_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).cuda()

                # Process each batch
                outputs = self.model(X_batch).detach().cpu().numpy()
                predictions.append(outputs)

                # Explicitly delete tensors to free up GPU memory
                del X_batch, outputs
                torch.cuda.empty_cache()  # Clear memory cache to prevent CUDA out of memory errors

        # Concatenate all batch outputs
        return np.concatenate(predictions, axis=0)

    def evaluate(self, X, y, batch_size=32):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Assuming X and y are numpy arrays or similar, batch processing is done without prior conversion
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # Convert slices of arrays to tensors directly and move to GPU
                X_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).cuda()
                y_batch = torch.tensor(y[i:i + batch_size], dtype=torch.float32).cuda()

                # Forward pass
                outputs = self.model(X_batch)
                loss = nn.MSELoss()(outputs,
                                    y_batch).item()  # Use .item() to get the Python number from a tensor with one element
                total_loss += loss
                n_batches += 1

                # Explicitly delete tensors to free up GPU memory
                del X_batch, y_batch, outputs
                torch.cuda.empty_cache()  # Clear memory cache to prevent CUDA out of memory error

        # Calculate average loss over all batches
        average_loss = total_loss / n_batches if n_batches > 0 else 0
        return average_loss

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
        self.model.cuda()
        self.model.eval()

class LSTMModel(BaseRNNModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, 'LSTM')

class GRUModel(BaseRNNModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, 'GRU')

class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, padding):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim,output_dim, num_channels, kernel_size):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TCNBlock(input_dim if i == 0 else num_channels[i - 1],
                         num_channels[i],
                         kernel_size,
                         dilation,
                         padding))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert to (batch, channels, length)
        out = self.network(x)
        out = out[:, :, -1]  # Select last valid output
        return self.linear(out)

class BaseTCNModel(Model):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.criterion = nn.L1Loss()
        self.optimizer = None
        self.epochs = None
        self.batch_size = None
        self.num_channels = None
        self.kernel_size = None
        self.model = None

    def set_params(self, lr=0.001, epochs=100, batch_size=32,num_channels=[16,32,64],kernel_size=3):
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.model = TemporalConvNet(self.input_dim,self.output_dim,self.num_channels, self.kernel_size).cuda()
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
                if outputs.shape==labels.shape:
                    loss = 0.5 * self.criterion(outputs, labels) + 0.5 * nn.MSELoss()(outputs,labels)
                else:
                    loss = 0.5 * self.criterion(outputs, labels.unsqueeze(-1)) + 0.5 * nn.MSELoss()(outputs,
                                                                                                    labels.unsqueeze(-1))
                loss.backward()
                self.optimizer.step()
        del dataset,X,y,loss,outputs
        torch.cuda.empty_cache()
    def predict(self, X, batch_size=32):
        self.model.eval()
        predictions = []

        # Convert the entire array to a tensor first to avoid multiple GPU transfers
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                # Process each batch
                X_batch = X_tensor[i:i + batch_size]
                outputs = self.model(X_batch).detach().cpu().numpy()
                predictions.append(outputs)
        del X_tensor,outputs
        torch.cuda.empty_cache()

        # Concatenate all batch outputs
        return np.concatenate(predictions, axis=0)

    def evaluate(self, X, y, batch_size=32):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Assuming X and y are numpy arrays or similar, batch processing is done without prior conversion
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # Convert slices of arrays to tensors directly and move to GPU
                X_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).cuda()
                y_batch = torch.tensor(y[i:i + batch_size], dtype=torch.float32).cuda()

                # Forward pass
                outputs = self.model(X_batch)
                loss = nn.MSELoss()(outputs,
                                    y_batch).item()  # Use .item() to get the Python number from a tensor with one element
                total_loss += loss
                n_batches += 1

                # Explicitly delete tensors to free up GPU memory
                del X_batch, y_batch, outputs
                torch.cuda.empty_cache()  # Clear memory cache to prevent CUDA out of memory error

        # Calculate average loss over all batches
        average_loss = total_loss / n_batches if n_batches > 0 else 0
        return average_loss

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
        self.model.cuda()
        self.model.eval()


class FreTS(Model):
    class FresTSModel(nn.Module):
        def __init__(self,input_length,input_size,output_size,embed_size,
                                      hidden_size,channel_independence):
            super(FreTS.FresTSModel,self).__init__()
            self.embed_size = embed_size  # embed_size
            self.hidden_size = hidden_size  # hidden_size
            self.pre_length = output_size
            self.feature_size = input_size  # channels
            self.seq_length = input_length
            self.channel_independence = channel_independence
            self.sparsity_threshold = 0.01
            self.scale = 0.02
            self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
            self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

            self.fc = nn.Sequential(
                nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.pre_length)
            )

        # dimension extension
        def tokenEmb(self, x):
            # x: [Batch, Input length, Channel]
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(3)
            # N*T*1 x 1*D = N*T*D
            y = self.embeddings
            return x * y

        # frequency temporal learner
        def MLP_temporal(self, x, B, N, L):
            # [B, N, T, D]
            x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
            y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
            x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
            return x

        # frequency channel learner
        def MLP_channel(self, x, B, N, L):
            # [B, N, T, D]
            x = x.permute(0, 2, 1, 3)
            # [B, T, N, D]
            x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
            y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
            x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
            x = x.permute(0, 2, 1, 3)
            # [B, N, T, D]
            return x

        def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
            o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                                  device=x.device)
            o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                                  device=x.device)

            o1_real = F.relu(
                torch.einsum('bijd,dd->bijd', x.real, r) -
                torch.einsum('bijd,dd->bijd', x.imag, i) +
                rb
            )

            o1_imag = F.relu(
                torch.einsum('bijd,dd->bijd', x.imag, r) +
                torch.einsum('bijd,dd->bijd', x.real, i) +
                ib
            )

            y = torch.stack([o1_real, o1_imag], dim=-1)
            y = F.softshrink(y, lambd=self.sparsity_threshold)
            y = torch.view_as_complex(y)
            return y

        def forward(self, x):
            # x: [Batch, Input length, Channel]
            B, T, N = x.shape
            # embedding x: [B, N, T, D]
            x = self.tokenEmb(x)
            bias = x
            # [B, N, T, D]
            if self.channel_independence == '1':
                x = self.MLP_channel(x, B, N, T)
            # [B, N, T, D]
            x = self.MLP_temporal(x, B, N, T)
            x = x + bias
            x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
            x = x[:,:,-1]
            return x

    def __init__(self,input_length,input_size,output_size):
        self.input_length = input_length
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = None
        self.hidden_size = None
        self.channel_independence = None
        self.epochs = None
        self.batch_size = None
        self.optimizer = None
        self.criterion = nn.L1Loss()

    def set_params(self, lr=0.001, epochs=100, batch_size=32,embed_size=128,hidden_size=256,channel_independence='1'):
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channel_independence = channel_independence
        self.model = self.FresTSModel(self.input_length, self.input_size, self.output_size,self.embed_size,
                                      self.hidden_size,self.channel_independence).cuda()
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
                if outputs.dim() == labels.dim():
                    loss = 0.5 * self.criterion(outputs, labels) + 0.5 * nn.MSELoss()(outputs, labels)
                else:
                    loss = 0.5 * self.criterion(outputs, labels.unsqueeze(-1)) + 0.5 * nn.MSELoss()(outputs,
                                                                                                    labels.unsqueeze(
                                                                                                        -1))

                loss.backward()
                self.optimizer.step()

    def predict(self, X, batch_size=32):
        self.model.eval()
        predictions = []

        # Avoid converting the entire array to a GPU tensor at once
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # Convert slices of arrays to tensors directly and move to GPU
                X_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).cuda()

                # Process each batch
                outputs = self.model(X_batch).detach().cpu().numpy()
                predictions.append(outputs)

                # Explicitly delete tensors to free up GPU memory
                del X_batch, outputs
                torch.cuda.empty_cache()  # Clear memory cache to prevent CUDA out of memory errors

        # Concatenate all batch outputs
        return np.concatenate(predictions, axis=0)

    def evaluate(self, X, y, batch_size=32):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Assuming X and y are numpy arrays or similar, batch processing is done without prior conversion
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # Convert slices of arrays to tensors directly and move to GPU
                X_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).cuda()
                y_batch = torch.tensor(y[i:i + batch_size], dtype=torch.float32).cuda()

                # Forward pass
                outputs = self.model(X_batch)
                loss = nn.MSELoss()(outputs,
                                      y_batch).item()  # Use .item() to get the Python number from a tensor with one element
                total_loss += loss
                n_batches += 1

                # Explicitly delete tensors to free up GPU memory
                del X_batch, y_batch, outputs
                torch.cuda.empty_cache()  # Clear memory cache to prevent CUDA out of memory error

        # Calculate average loss over all batches
        average_loss = total_loss / n_batches if n_batches > 0 else 0
        return average_loss

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
        self.model.eval()

param_grid = {
    'LSTM':
    {
    'hidden_dim': [50],
    'num_layers': [1,2],
    'lr': [0.01],
    'batch_size': [ 32],
    'epochs': [50]},
    'GRU':
{
    'hidden_dim': [50],
    'num_layers': [2],
    'lr': [0.01],
    'batch_size': [ 32],
    'epochs': [50]},
    'TCN':
{
    'kernel_size':[5,10,20],
    'num_channels': [[32,64,128],[64,128,256]],
    'lr': [0.0001],
    'batch_size': [256],
    'epochs': [10]},
    'FreTS':
{
    'channel_independence':['1'],
    'hidden_size': [512,256],
    'embed_size': [256,128],
    'lr': [0.0001,0.001],
    'batch_size': [256],
    'epochs': [10,20,50]},
    'ITransformer':
{
    'lr': [0.0001],
    'batch_size': [256],
    'epochs': [10,20]},
}
