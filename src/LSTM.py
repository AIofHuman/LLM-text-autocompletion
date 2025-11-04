import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class LSTMAutocomplete(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization with forget gate bias=1"""
        # Embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Output layer
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x, attention_mask=None, hidden=None):
        # x shape: (batch_size, seq_length)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # lstm_out: (batch_size, seq_length, hidden_dim)
        
        # Use attention mask to find last non-padded token
        if attention_mask is not None:
            # Get the index of the last real token (not padding)
            lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
            # Use advanced indexing to get the last real output for each sequence
            last_output = lstm_out[torch.arange(lstm_out.size(0)), lengths]
        else:
            # Fallback to just taking the last position
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Final output
        output = self.fc(last_output)  # (batch_size, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)