import torch
from torch import nn
from skorch import NeuralNetRegressor
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


net = NeuralNetRegressor(
    module=LSTMModel,
    module__input_dim=1,  # 例如，使用单变量时间序列
    module__hidden_dim=30,
    module__output_dim=1,
    module__num_layers=1,
    criterion=torch.nn.MSELoss,
    max_epochs=50,
    lr=0.01,
    batch_size=32,
    optimizer=torch.optim.Adam
)