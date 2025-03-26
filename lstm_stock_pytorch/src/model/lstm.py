import torch
import torch.nn as nn
import numpy as np

class LSTMPredictor(nn.Module):
    """
    LSTM model for stock prediction with enhanced architecture.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, 
                 bidirectional=False, use_attention=True):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Determine output size based on bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_size // 2, 1)
            )
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Final output layer with Tanh activation for position sizing (-1 to 1)
        self.fc_out = nn.Linear(hidden_size // 2, 1)
        self.tanh = nn.Tanh()
        
        # Batch normalization to improve training stability
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights - use orthogonal initialization
                    nn.init.orthogonal_(param)
                else:
                    # FC layers - use Xavier initialization
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def apply_attention(self, lstm_output):
        """Apply attention mechanism to LSTM output."""
        # Calculate attention weights
        attn_weights = self.attention(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        
        if self.use_attention:
            # Apply attention to focus on important timesteps
            context = self.apply_attention(lstm_out)
        else:
            # Use last timestep
            context = lstm_out[:, -1, :]
        
        # Fully connected layers with residual connections and batch normalization
        out = self.fc1(context)
        if batch_size > 1:  # BatchNorm1d requires batch size > 1
            out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        if batch_size > 1:
            out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        
        # Final output with Tanh activation for position sizing (-1 to 1)
        out = self.fc_out(out)
        out = self.tanh(out)
        
        return out
