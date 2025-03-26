import nbformat as nbf
import os

# 创建一个新的notebook
nb = nbf.v4.new_notebook()

# 添加标题和介绍
intro_cell = nbf.v4.new_markdown_cell('''
# LSTM Stock Prediction - Standalone Version with GPU Support

This notebook contains a complete implementation of LSTM stock prediction with sequential validation, optimized for GPU acceleration in Google Colab.

## Features
- GPU-accelerated training
- Download stock data from Yahoo Finance
- Prepare data with technical indicators
- Train LSTM model with custom loss function
- Sequential validation with walk-forward testing
- Market regime analysis
- Visualizations and performance metrics

## GPU Setup in Colab
1. Go to Runtime > Change runtime type
2. Select "GPU" as Hardware accelerator
3. Click "Save"

## Setup and Requirements
First, let's install the required packages and configure GPU support.
''')

# 安装包的代码单元
install_cell = nbf.v4.new_code_cell('''
# Install required packages
!pip install -q yfinance pandas matplotlib seaborn scikit-learn torch tqdm
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
''')

# 导入库的代码单元
import_cell = nbf.v4.new_code_cell('''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import datetime
from datetime import timedelta
from tqdm.notebook import tqdm
import math
import random
import time
import warnings
import seaborn as sns
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# GPU Configuration and Memory Management
def setup_gpu():
    """Configure GPU settings and memory management."""
    if torch.cuda.is_available():
        # Get GPU properties
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        
        print(f"Using GPU: {gpu_name}")
        print(f"Total GPU Memory: {gpu_memory:.2f} GB")
        
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocator settings
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
    else:
        print("No GPU available. Using CPU.")
        return False

# Initialize GPU
device = torch.device("cuda" if setup_gpu() else "cpu")
print(f"Using device: {device}")

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("LSTM Stock Prediction - Standalone Version with GPU Support")
print("==========================================================")
''')

# GPU内存管理函数
gpu_memory_cell = nbf.v4.new_code_cell('''
def manage_gpu_memory():
    """Monitor and manage GPU memory usage."""
    if torch.cuda.is_available():
        # Get current GPU memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"GPU Memory Allocated: {allocated:.2f} GB")
        print(f"GPU Memory Cached: {cached:.2f} GB")
        
        # Clear cache if memory usage is high
        if allocated > 0.8:  # If more than 80% memory is used
            torch.cuda.empty_cache()
            gc.collect()
            print("Cleared GPU cache")
            
def to_device(data, device):
    """Move data to specified device (CPU/GPU)."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)
''')

# LSTM模型定义的标题
model_title = nbf.v4.new_markdown_cell('''
## LSTM Model Definition

Now, let's define the LSTM model and the custom loss function with GPU optimization.
''')

# LSTM模型定义代码
lstm_model_cell = nbf.v4.new_code_cell('''
# Define the LSTM model
class LSTMPredictor(nn.Module):
    """
    LSTM model for stock prediction with enhanced architecture and GPU optimization.
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
        
        # LSTM layer with GPU optimization
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
''')

# 自定义损失函数代码
loss_function_cell = nbf.v4.new_code_cell('''
# Define custom loss functions
class KellyDrawdownLoss(nn.Module):
    """
    Combined Kelly Criterion and Drawdown optimization with GPU support.
    
    Combines:
    1. Kelly Criterion for position sizing
    2. Drawdown penalty
    3. Smoothness penalty for equity curve
    """
    def __init__(self, alpha=0.5, max_leverage=2.0, dd_weight=1.0, smoothness_weight=0.5):
        super().__init__()
        self.alpha = alpha  # Weight between Kelly position and return maximization
        self.max_leverage = max_leverage
        self.dd_weight = dd_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, preds, targets):
        # Ensure inputs are on the same device
        preds = preds.to(preds.device)
        targets = targets.to(preds.device)
        
        # Portfolio returns based on predicted positions and actual returns
        portfolio_returns = preds * targets
        
        # Kelly criterion component
        mu = torch.mean(portfolio_returns)
        sigma = torch.std(portfolio_returns) + 1e-6
        
        # Optimal Kelly fraction
        kelly_f = torch.clamp(mu / (sigma**2 + 1e-6), -self.max_leverage, self.max_leverage)
        
        # Position sizing loss (MSE to optimal Kelly fraction)
        position_loss = torch.mean((preds - kelly_f)**2)
        
        # Return maximization (negative mean return)
        return_loss = -mu
        
        # Drawdown calculation
        cum_returns = torch.cumsum(portfolio_returns, dim=0)
        running_max = torch.cummax(cum_returns, dim=0)[0]
        drawdowns = running_max - cum_returns
        max_drawdown = torch.max(drawdowns)
        
        # Smoothness calculation - penalize large swings in returns
        if len(portfolio_returns) > 1:
            return_changes = torch.diff(portfolio_returns, dim=0)
            smoothness_loss = torch.std(return_changes)
        else:
            smoothness_loss = torch.tensor(0.0, device=preds.device)
        
        # Combine losses with weights
        kelly_component = self.alpha * position_loss + (1 - self.alpha) * return_loss
        drawdown_component = max_drawdown * self.dd_weight
        smoothness_component = smoothness_loss * self.smoothness_weight
        
        total_loss = kelly_component + drawdown_component + smoothness_component
        
        return total_loss
''')

# 数据处理标题
data_processing_title = nbf.v4.new_markdown_cell('''
## Data Processing Class

Now, let's define the data processing class that will handle downloading and preparing the stock data with GPU support.
''')

# 模型训练标题  
model_training_title = nbf.v4.new_markdown_cell('''
## Model Training Class

Next, let's define the class for training the LSTM model with GPU optimization.
''')

# 主函数和使用示例标题
main_function_title = nbf.v4.new_markdown_cell('''
## Example Usage

Now, let's demonstrate the usage of the LSTM stock prediction framework with GPU acceleration.
''')

# 向notebook添加所有单元格
nb.cells = [
    intro_cell,
    install_cell,
    import_cell,
    gpu_memory_cell,
    model_title,
    lstm_model_cell,
    loss_function_cell,
    data_processing_title,
    model_training_title,
    main_function_title
]

# 设置notebook的元数据
nb.metadata = {
    'colab': {
        'name': 'lstm_stock_colab_standalone.ipynb',
        'provenance': [],
        'collapsed_sections': []
    },
    'kernelspec': {
        'name': 'python3',
        'display_name': 'Python 3'
    },
    'language_info': {
        'name': 'python'
    },
    'accelerator': 'GPU'
}

# 将notebook写入文件
with open('lstm_stock_colab_standalone.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook created: lstm_stock_colab_standalone.ipynb") 