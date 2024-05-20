import torch
import torch.nn as nn
import numpy as np

from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from .models import Model
class ItransformerModel(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    lr=0.001, epochs=100, batch_size=32, d_model=512, n_heads=8, e_layers=3, d_ff=2048,
                   dropout=0.1, activation='gelu', factor=5, embed='fixed', freq='h'
    """

    def __init__(self, seq_len,pred_len,output_attention,use_norm,class_strategy,d_model=512,embed='fixed',freq='h',dropout=0.1,
                 factor=5,n_heads=8,d_ff=2048,activation='gelu',e_layers=3):
        super(ItransformerModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq,
                                                    dropout)
        self.class_strategy = class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class ITransformer(Model):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, input_dim, output_dim, seq_len, pred_len, output_attention=False, use_norm=False, class_strategy=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.class_strategy = class_strategy
        self.criterion = nn.L1Loss()
        self.optimizer = None
        self.epochs = None
        self.batch_size = None
        self.model = None

    def set_params(self, lr=0.001, epochs=100, batch_size=32, d_model=512, n_heads=8, e_layers=3, d_ff=2048,
                   dropout=0.1, activation='gelu', factor=5, embed='fixed', freq='h'):
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.factor = factor
        self.embed = embed
        self.freq = freq

        self.model = ItransformerModel(
            seq_len=self.seq_len, pred_len=self.pred_len, output_attention=self.output_attention, use_norm=self.use_norm,
            d_model=d_model,embed=embed, freq=freq, dropout=dropout, class_strategy=self.class_strategy,
            factor=factor, n_heads=n_heads,d_ff= d_ff, activation=activation, e_layers=e_layers
        ).cuda()

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
                if outputs.shape == labels.shape:
                    loss = 0.5 * self.criterion(outputs, labels) + 0.5 * nn.MSELoss()(outputs, labels)
                else:
                    loss = 0.5 * self.criterion(outputs, labels.unsqueeze(-1)) + 0.5 * nn.MSELoss()(outputs,
                                                                                                    labels.unsqueeze(
                                                                                                        -1))
                loss.backward()
                self.optimizer.step()
        del dataset, X, y, loss, outputs
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
        del X_tensor, outputs
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
        average_loss = total_loss / n_batches if n_batches >0 else 0
        return average_loss

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
        self.model.cuda()
        self.model.eval()
