import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device
        self.pe = self._get_positional_encoding()

    def _get_positional_encoding(self):
        position = torch.arange(0, self.max_len, device = self.device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2, device = self.device).float() * -(torch.log(torch.tensor(10000.0)) / self.hidden_dim))
        pe = torch.zeros(self.max_len, self.hidden_dim, device = self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class TransformerModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_heads, 
                 num_layers, 
                 hidden_dim, 
                 max_len, 
                 dropout,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len, device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        mask = mask.permute(1,0)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = torch.mean(x, dim=1)  # Aggregate by averaging over the sequence length
        # x = x[:, 0, :]
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)
        x = x*~mask # no effect
        out = self.fc1(x)
        out = self.relu(out)
        out = torch.mean(out, dim=1)
        out = self.fc2(out)
        return out
    
class MLP_gpt(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, mask):
        out = self.fc1(x.squeeze(-1))
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class LR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x, mask):
        out = self.linear(x.squeeze(-1))
        return out


class MLP_fix(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x, mask):
        out = self.fc1(x.squeeze(-1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
