import torch
import torch.nn as nn

class TemporalNetwork(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256, num_layers=2, dropout=0.2, seq_len=50):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # LSTM-based temporal network
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Attention layer for better long-term dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        # Reconstructor to predict original CAN frame from features
        self.reconstructor = nn.Linear(hidden_dim, self.input_dim)
    
    def forward(self, x):
        # x: (B, seq_len, input_dim)
        batch_size = x.size(0)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, seq_len, hidden_dim)
        
        # Self-attention on the last timestep
        last_timestep = lstm_out[:, -1:, :]  # (B, 1, hidden_dim)
        attn_out, _ = self.attention(
            last_timestep, lstm_out, lstm_out
        )  # (B, 1, hidden_dim)
        
        # Project to final feature vector
        features = self.proj(attn_out.squeeze(1))  # (B, hidden_dim)

        # Reconstruct CAN-frame-sized output for temporal prediction tasks
        recon = self.reconstructor(features)  # (B, input_dim)

        return features, recon


class TransformerTemporalNetwork(nn.Module):
    """Alternative: Transformer-based temporal network"""
    def __init__(self, input_dim=12, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.embedding(x)  # (B, seq_len, d_model)
        x = self.transformer(x)  # (B, seq_len, d_model)
        # Take the last timestep
        features = self.proj(x[:, -1, :])  # (B, d_model)
        return features