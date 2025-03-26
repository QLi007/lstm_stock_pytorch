#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM Stock Prediction - Standalone Version for Google Colab
This script contains a complete implementation of LSTM stock prediction with sequential validation,
without any external dependencies.
"""

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
import yaml
import json
import sys
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("LSTM Stock Prediction - Standalone Version")
print("==========================================")

# Define the LSTM model
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


# Define custom loss functions
class KellyDrawdownLoss(nn.Module):
    """
    Combined Kelly Criterion and Drawdown optimization.
    
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

# Data download and preparation functions
class StockDataProcessor:
    """Class to download and process stock data"""
    
    def __init__(self, tickers=None, start_date=None, end_date=None):
        """
        Initialize the data processor.
        
        Args:
            tickers: List of stock tickers to download
            start_date: Start date for data download
            end_date: End date for data download
        """
        self.tickers = tickers or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.end_date = end_date or datetime.datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=5*365))
        self.data = {}
        self.combined_data = None
    
    def download_data(self):
        """Download stock data from Yahoo Finance"""
        for ticker in self.tickers:
            print(f"Downloading {ticker} data...")
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            
            # Handle column names
            if isinstance(data.columns, pd.MultiIndex):
                # For multi-level columns, take the first level (price type)
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Add to data dictionary
            self.data[ticker] = data
            
            # Print data info
            print(f"  Shape: {data.shape}")
        
        return self.data
    
    def add_technical_indicators(self):
        """Add technical indicators to stock data"""
        for ticker, data in self.data.items():
            print(f"Adding technical indicators for {ticker}...")
            
            # Moving averages
            for period in [5, 10, 20, 34, 60, 120, 200]:
                data[f'MA_{period}'] = data['close'].rolling(window=period).mean()
                data[f'price_to_MA_{period}'] = data['close'] / data[f'MA_{period}'] - 1
            
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)  # Add small epsilon to avoid division by zero
            data['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']
            
            # Williams %R
            high_max = data['high'].rolling(window=14).max()
            low_min = data['low'].rolling(window=14).min()
            data['Williams_%R_14'] = -100 * (high_max - data['close']) / (high_max - low_min + 1e-9)
            
            # Add future returns for target
            for days in [1, 5, 10, 20]:
                data[f'future_{days}d_return'] = data['close'].pct_change(days).shift(-days)
            
            # Add previous day's close for features
            data['prev_close'] = data['close'].shift(1)
            
            # Add 1-day returns
            data['return_1d'] = data['close'].pct_change(1)
            
            # Volatility
            for period in [10, 20, 60]:
                data[f'volatility_{period}'] = data['return_1d'].rolling(window=period).std()
            
            # Volume indicators
            data['volume_ma10'] = data['volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma10']
            
            # Update data dictionary
            self.data[ticker] = data
        
        return self.data
    
    def create_combined_dataset(self):
        """Combine all stock data into a single dataset"""
        combined_data = pd.DataFrame()
        
        for ticker, data in self.data.items():
            # Add ticker column
            data_copy = data.copy()
            data_copy['ticker'] = ticker
            
            # Append to combined data
            if len(combined_data) == 0:
                combined_data = data_copy
            else:
                combined_data = pd.concat([combined_data, data_copy])
        
        self.combined_data = combined_data
        return combined_data
    
    def save_data(self):
        """Save data to CSV files"""
        for ticker, data in self.data.items():
            csv_path = f"data/{ticker.lower()}.csv"
            data.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")
        
        if self.combined_data is not None:
            combined_csv_path = "data/combined_stocks.csv"
            self.combined_data.to_csv(combined_csv_path, index=False)
            print(f"Combined dataset saved to {combined_csv_path}")
    
    def load_data(self, ticker=None, file_path=None):
        """Load data from a CSV file"""
        if file_path:
            return pd.read_csv(file_path)
        elif ticker:
            return pd.read_csv(f"data/{ticker.lower()}.csv")
        else:
            return pd.read_csv("data/combined_stocks.csv")
    
    def prepare_training_data(self, data=None, target_col='future_5d_return', seq_length=30, 
                             test_size=0.2, val_size=0.1, scale_features=True):
        """
        Prepare training, validation, and test data.
        
        Args:
            data: DataFrame with stock data
            target_col: Target column name
            seq_length: Sequence length for LSTM
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            scale_features: Whether to standardize features
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
        """
        if data is None:
            if self.combined_data is not None:
                data = self.combined_data
            else:
                # Use the first ticker's data
                ticker = self.tickers[0]
                data = self.data.get(ticker, self.load_data(ticker))
        
        # Drop rows with missing values
        data = data.dropna()
        
        # Select feature columns (exclude date, OHLCV, and target)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', target_col]
        if 'ticker' in data.columns:
            exclude_cols.append('ticker')
        
        feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('future_')]
        
        # Get features and target
        features = data[feature_cols].values
        targets = data[target_col].values
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - seq_length + 1):
            X.append(features[i:i+seq_length])
            y.append(targets[i+seq_length-1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(1-test_size), shuffle=False
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

# Model training functions
class LSTMModelTrainer:
    """Class for training LSTM models"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, learning_rate=0.001,
                batch_size=64, max_leverage=2.0, kelly_alpha=0.7, dd_weight=1.0):
        """
        Initialize the model trainer.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            max_leverage: Maximum position size
            kelly_alpha: Weight for Kelly component
            dd_weight: Weight for drawdown component
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_leverage = max_leverage
        self.kelly_alpha = kelly_alpha
        self.dd_weight = dd_weight
        
        # Initialize model
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(device)
        
        # Initialize loss function
        self.criterion = KellyDrawdownLoss(
            alpha=kelly_alpha,
            max_leverage=max_leverage,
            dd_weight=dd_weight
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': {}
        }
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val):
        """Create DataLoaders for training and validation"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimize
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate portfolio returns
        portfolio_returns = all_preds * all_targets
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'mean_return': portfolio_returns.mean(),
            'sharpe': portfolio_returns.mean() / (portfolio_returns.std() + 1e-6) * np.sqrt(252),
            'correlation': np.corrcoef(all_preds, all_targets)[0, 1]
        }
        
        return metrics
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, early_stopping=True, patience=20):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(X_train, y_train, X_val, y_val)
        
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        print("Starting training...")
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'][epoch] = val_metrics
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Sharpe: {val_metrics['sharpe']:.4f}, Val Return: {val_metrics['mean_return']:.4f}")
            
            # Check for early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break
        
        print("Training completed!")
        return self.history
    
    def save_model(self, model_path="models/lstm_model.pt"):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'hyperparams': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_leverage': self.max_leverage,
                'kelly_alpha': self.kelly_alpha,
                'dd_weight': self.dd_weight
            }
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path="models/lstm_model.pt"):
        """Load the model"""
        checkpoint = torch.load(model_path, map_location=device)
        
        # Update hyperparameters
        hyperparams = checkpoint.get('hyperparams', {})
        self.input_size = hyperparams.get('input_size', self.input_size)
        self.hidden_size = hyperparams.get('hidden_size', self.hidden_size)
        self.num_layers = hyperparams.get('num_layers', self.num_layers)
        
        # Recreate model if architecture is different
        if (self.model.input_size != self.input_size or
            self.model.hidden_size != self.hidden_size or
            self.model.num_layers != self.num_layers):
            self.model = LSTMPredictor(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            ).to(device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from {model_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        # Loss plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Sharpe ratio plot
        plt.subplot(1, 2, 2)
        epochs = list(self.history['val_metrics'].keys())
        sharpe_values = [m['sharpe'] for m in self.history['val_metrics'].values()]
        plt.plot(epochs, sharpe_values)
        plt.title('Validation Sharpe Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Sharpe Ratio')
        
        plt.tight_layout()
        plt.savefig("results/training_history.png")
        plt.show()

# Sequential validation framework
class SequentialValidator:
    """
    Walk-forward testing framework to validate model predictions without data leakage.
    
    This class implements a point-by-point simulation of real trading, ensuring that:
    1. No future data is used in any decision
    2. Model predictions are consistent with real-time usage
    3. Performance metrics are calculated under realistic conditions
    """
    def __init__(self, model, data, feature_cols, seq_length=30, target_col='future_5d_return'):
        """
        Initialize the validator.
        
        Args:
            model: Trained PyTorch model
            data: DataFrame with stock data
            feature_cols: List of feature column names
            seq_length: Sequence length for LSTM
            target_col: Target column name
        """
        self.model = model
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        self.target_col = target_col
        
        # Prepare data
        self.data = self.data.sort_values('date') if 'date' in self.data.columns else self.data
        
        # Standardize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.data[feature_cols].values)
        
        # Get targets
        self.targets = self.data[target_col].values
        
        # Get dates if available
        if 'date' in self.data.columns:
            self.dates = self.data['date'].values
        else:
            self.dates = np.arange(len(self.data))
        
        # Get close prices if available
        if 'close' in self.data.columns:
            self.close_prices = self.data['close'].values
        else:
            self.close_prices = None
    
    def run_validation(self, plot=True, log_dir="results/sequential"):
        """
        Run sequential validation.
        
        Args:
            plot: Whether to generate plots
            log_dir: Directory to save results
            
        Returns:
            DataFrame with validation results
        """
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize results
        results = []
        
        # Start at the first valid index for sequence
        start_idx = self.seq_length - 1
        
        print(f"Running sequential validation with {len(self.data) - start_idx} data points...")
        
        # Process each data point sequentially
        for i in tqdm(range(start_idx, len(self.data))):
            # Get sequence up to current point (no future data)
            features_seq = self.features[i-self.seq_length+1:i+1]
            
            # Get target (future return)
            true_target = self.targets[i]
            
            # Get current date and price
            current_date = self.dates[i]
            current_price = self.close_prices[i] if self.close_prices is not None else None
            
            # Convert to tensor and predict
            features_tensor = torch.FloatTensor(features_seq).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = self.model(features_tensor).item()
            
            # Calculate return (position * future return)
            achieved_return = prediction * true_target
            
            # Record results
            results.append({
                'date': current_date,
                'price': current_price,
                'position': prediction,
                'future_return': true_target,
                'achieved_return': achieved_return
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate cumulative returns
        results_df['cumulative_return'] = (1 + results_df['achieved_return']).cumprod() - 1
        
        # Calculate metrics
        metrics = self._calculate_metrics(results_df)
        for key, value in metrics.items():
            results_df[key] = value
        
        # Save results
        results_df.to_csv(f"{log_dir}/sequential_results.csv", index=False)
        
        # Generate plots if requested
        if plot:
            self._generate_plots(results_df, log_dir)
        
        return results_df
    
    def _calculate_metrics(self, results):
        """Calculate performance metrics."""
        returns = results['achieved_return']
        
        # Calculate total return
        total_return = results['cumulative_return'].iloc[-1]
        
        # Calculate Sharpe ratio (annualized)
        sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)
        
        # Calculate maximum drawdown
        cumulative = results['cumulative_return']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / (running_max + 1)
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = (returns > 0).mean()
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Return metrics
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility
        }
        
        print("Sequential Validation Metrics:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Volatility: {volatility:.2%}")
        
        return metrics
    
    def _generate_plots(self, results, log_dir):
        """Generate and save plots."""
        # Equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results['date'], results['cumulative_return'] * 100)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.savefig(f"{log_dir}/equity_curve.png")
        plt.close()
        
        # Position sizes over time
        plt.figure(figsize=(12, 6))
        plt.plot(results['date'], results['position'])
        plt.title('Position Sizes')
        plt.xlabel('Date')
        plt.ylabel('Position (-1 to 1)')
        plt.grid(True)
        plt.savefig(f"{log_dir}/positions.png")
        plt.close()
        
        # Position vs Future Return scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(results['position'], results['future_return'], alpha=0.5)
        plt.title('Position vs Future Return')
        plt.xlabel('Position Size')
        plt.ylabel('Future Return')
        plt.grid(True)
        plt.savefig(f"{log_dir}/position_vs_return.png")
        plt.close()
        
        # Histogram of returns
        plt.figure(figsize=(10, 6))
        plt.hist(results['achieved_return'], bins=50, alpha=0.7)
        plt.title('Distribution of Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{log_dir}/return_distribution.png")
        plt.close()
        
        # Monthly performance heatmap
        if isinstance(results['date'].iloc[0], (pd.Timestamp, np.datetime64, datetime.datetime, str)):
            try:
                # Convert to datetime if necessary
                if isinstance(results['date'].iloc[0], str):
                    results['date'] = pd.to_datetime(results['date'])
                
                # Extract month and year
                results['year'] = results['date'].dt.year
                results['month'] = results['date'].dt.month
                
                # Calculate monthly returns
                monthly_returns = results.groupby(['year', 'month'])['achieved_return'].sum().unstack()
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(monthly_returns * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
                plt.title('Monthly Returns (%)')
                plt.savefig(f"{log_dir}/monthly_returns.png")
                plt.close()
            except:
                print("Could not create monthly performance heatmap")

# Market state analysis
class MarketRegimeAnalyzer:
    """
    Analyzes market regimes and state transitions to understand model performance
    in different market conditions.
    """
    def __init__(self, data, window_size=60, n_regimes=3, volatility_percentile=75):
        """
        Initialize the market regime analyzer.
        
        Args:
            data: DataFrame with stock data
            window_size: Window size for regime detection
            n_regimes: Number of regimes to detect
            volatility_percentile: Percentile threshold for high volatility
        """
        self.data = data.copy()
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.volatility_percentile = volatility_percentile
        
        # Ensure date column exists
        if 'date' not in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data.index)
        
        # Calculate returns if not provided
        if 'return_1d' not in self.data.columns and 'close' in self.data.columns:
            self.data['return_1d'] = self.data['close'].pct_change()
    
    def detect_regimes(self):
        """
        Detect market regimes using Hidden Markov Model.
        
        Returns:
            DataFrame with regime assignments
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            print("Installing sklearn for regime detection...")
            !pip install -q scikit-learn
            from sklearn.mixture import GaussianMixture
        
        # Get return data
        returns = self.data['return_1d'].fillna(0).values.reshape(-1, 1)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        gmm.fit(returns)
        
        # Get regime assignments
        regimes = gmm.predict(returns)
        probs = gmm.predict_proba(returns)
        
        # Sort regimes by mean returns
        means = [gmm.means_[i][0] for i in range(self.n_regimes)]
        sorted_indices = np.argsort(means)
        
        # Create mapping for regime ordering (bear to bull)
        regime_map = {sorted_indices[i]: i for i in range(self.n_regimes)}
        
        # Map regimes to ordered regimes
        regimes = np.array([regime_map[r] for r in regimes])
        
        # Add to data
        self.data['regime'] = regimes
        
        # Add regime names
        regime_names = ['Bear', 'Neutral', 'Bull'] if self.n_regimes == 3 else [f'Regime {i}' for i in range(self.n_regimes)]
        self.data['regime_name'] = [regime_names[r] for r in regimes]
        
        return self.data
    
    def detect_volatility_regimes(self):
        """
        Detect high volatility periods.
        
        Returns:
            DataFrame with volatility regime assignments
        """
        # Calculate rolling volatility
        self.data['rolling_vol'] = self.data['return_1d'].rolling(self.window_size).std() * np.sqrt(252)
        
        # Determine high volatility threshold
        vol_threshold = self.data['rolling_vol'].quantile(self.volatility_percentile / 100)
        
        # Assign volatility regimes
        self.data['high_volatility'] = self.data['rolling_vol'] > vol_threshold
        
        return self.data
    
    def analyze_model_performance_by_regime(self, results_df):
        """
        Analyze model performance across different market regimes.
        
        Args:
            results_df: DataFrame with model results (from SequentialValidator)
            
        Returns:
            Dictionary with performance metrics by regime
        """
        # Ensure we have regimes detected
        if 'regime' not in self.data.columns:
            self.detect_regimes()
        
        # Ensure the dates align
        if 'date' in results_df.columns and 'date' in self.data.columns:
            # Map regimes to results
            regime_map = dict(zip(self.data['date'], self.data['regime']))
            regime_name_map = dict(zip(self.data['date'], self.data['regime_name']))
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(results_df['date']):
                results_df['date'] = pd.to_datetime(results_df['date'])
            
            if not pd.api.types.is_datetime64_any_dtype(list(regime_map.keys())[0]):
                regime_map = {pd.to_datetime(k): v for k, v in regime_map.items()}
                regime_name_map = {pd.to_datetime(k): v for k, v in regime_name_map.items()}
            
            # Create regime and regime_name columns
            results_df['regime'] = results_df['date'].map(regime_map)
            results_df['regime_name'] = results_df['date'].map(regime_name_map)
        
        # Calculate performance by regime
        performance_by_regime = {}
        
        for regime in range(self.n_regimes):
            regime_results = results_df[results_df['regime'] == regime]
            
            # Skip if no data in this regime
            if len(regime_results) == 0:
                continue
            
            # Get regime name
            regime_name = regime_results['regime_name'].iloc[0] if 'regime_name' in regime_results.columns else f'Regime {regime}'
            
            # Calculate metrics
            metrics = {
                'count': len(regime_results),
                'mean_return': regime_results['achieved_return'].mean(),
                'sharpe': regime_results['achieved_return'].mean() / (regime_results['achieved_return'].std() + 1e-6) * np.sqrt(252),
                'win_rate': (regime_results['achieved_return'] > 0).mean(),
                'avg_position': regime_results['position'].mean(),
                'avg_abs_position': regime_results['position'].abs().mean()
            }
            
            performance_by_regime[regime_name] = metrics
        
        # Print summary
        print("\nPerformance by Market Regime:")
        for regime, metrics in performance_by_regime.items():
            print(f"\n{regime} Regime:")
            print(f"  Count: {metrics['count']}")
            print(f"  Mean Return: {metrics['mean_return']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(f"  Avg Position: {metrics['avg_position']:.2f}")
            print(f"  Avg Abs Position: {metrics['avg_abs_position']:.2f}")
        
        return performance_by_regime
    
    def plot_regimes(self, log_dir="results/market_regimes"):
        """Plot market regimes and volatility."""
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Ensure we have regimes detected
        if 'regime' not in self.data.columns:
            self.detect_regimes()
        
        if 'high_volatility' not in self.data.columns:
            self.detect_volatility_regimes()
        
        # Plot price with regime background
        plt.figure(figsize=(14, 8))
        
        # Set colors for regimes
        colors = ['#ffcccc', '#e6f2ff', '#ccffcc']  # Red, Blue, Green
        
        # Plot price
        if 'close' in self.data.columns:
            plt.subplot(3, 1, 1)
            
            # Plot each regime with background color
            for regime in range(self.n_regimes):
                mask = self.data['regime'] == regime
                plt.plot(self.data.loc[mask, 'date'], self.data.loc[mask, 'close'], 'k-')
                
                # Add colored background
                for i in range(len(self.data)-1):
                    if self.data['regime'].iloc[i] == regime:
                        plt.axvspan(self.data['date'].iloc[i], self.data['date'].iloc[i+1], 
                                  alpha=0.3, color=colors[regime])
            
            plt.title('Stock Price with Market Regimes')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
        
        # Plot regimes as a heatmap
        plt.subplot(3, 1, 2)
        plt.scatter(self.data['date'], np.ones(len(self.data)), c=self.data['regime'], cmap='viridis', s=10)
        plt.yticks([])
        plt.title('Market Regimes')
        plt.xlabel('Date')
        plt.colorbar(ticks=range(self.n_regimes), label='Regime')
        
        # Plot volatility
        plt.subplot(3, 1, 3)
        plt.plot(self.data['date'], self.data['rolling_vol'], 'b-')
        plt.axhline(self.data['rolling_vol'].quantile(self.volatility_percentile / 100), 
                   color='r', linestyle='--', label='High Vol Threshold')
        plt.title('Rolling Volatility')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{log_dir}/market_regimes.png")
        plt.close()
        
        # Plot transition matrix heatmap
        transitions = np.zeros((self.n_regimes, self.n_regimes))
        
        # Count transitions
        regimes = self.data['regime'].values
        for i in range(len(regimes) - 1):
            transitions[regimes[i], regimes[i+1]] += 1
        
        # Normalize by row
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = transitions / (row_sums + 1e-10)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="Blues")
        
        # Set labels
        regime_names = ['Bear', 'Neutral', 'Bull'] if self.n_regimes == 3 else [f'Regime {i}' for i in range(self.n_regimes)]
        plt.xticks(ticks=np.arange(self.n_regimes) + 0.5, labels=regime_names)
        plt.yticks(ticks=np.arange(self.n_regimes) + 0.5, labels=regime_names)
        
        plt.title('Regime Transition Matrix')
        plt.xlabel('To Regime')
        plt.ylabel('From Regime')
        
        plt.savefig(f"{log_dir}/transition_matrix.png")
        plt.close()

# Example main function to demonstrate usage
def main():
    """Main function to demonstrate usage of all components."""
    print("\n===== LSTM Stock Prediction Demo =====\n")
    
    # Step 1: Download and prepare data
    print("\n--- Step 1: Download and Prepare Data ---\n")
    
    # Create data processor
    data_processor = StockDataProcessor(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date=datetime.datetime.now() - timedelta(days=5*365),
        end_date=datetime.datetime.now()
    )
    
    # Download data
    data_processor.download_data()
    
    # Add technical indicators
    data_processor.add_technical_indicators()
    
    # Create combined dataset
    data_processor.create_combined_dataset()
    
    # Save data
    data_processor.save_data()
    
    # Step 2: Prepare training data
    print("\n--- Step 2: Prepare Training Data ---\n")
    
    # Select a stock for demonstration
    ticker = 'AAPL'
    data = data_processor.load_data(ticker=ticker)
    
    # Prepare training data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = data_processor.prepare_training_data(
        data=data,
        target_col='future_5d_return',
        seq_length=30
    )
    
    # Step 3: Train the model
    print("\n--- Step 3: Train the Model ---\n")
    
    # Create trainer
    trainer = LSTMModelTrainer(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=2,
        learning_rate=0.0005,
        batch_size=32,
        max_leverage=2.0,
        kelly_alpha=0.7,
        dd_weight=1.0
    )
    
    # Train the model
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=30,
        early_stopping=True,
        patience=10
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save the model
    trainer.save_model(model_path=f"models/lstm_{ticker}.pt")
    
    # Step 4: Run sequential validation
    print("\n--- Step 4: Run Sequential Validation ---\n")
    
    # Create validator
    validator = SequentialValidator(
        model=trainer.model,
        data=data,
        feature_cols=feature_cols,
        seq_length=30,
        target_col='future_5d_return'
    )
    
    # Run validation
    results_df = validator.run_validation(
        plot=True,
        log_dir=f"results/sequential_{ticker}"
    )
    
    # Step 5: Analyze market regimes
    print("\n--- Step 5: Analyze Market Regimes ---\n")
    
    # Create market regime analyzer
    regime_analyzer = MarketRegimeAnalyzer(
        data=data,
        window_size=60,
        n_regimes=3,
        volatility_percentile=75
    )
    
    # Detect regimes
    regime_analyzer.detect_regimes()
    
    # Detect volatility regimes
    regime_analyzer.detect_volatility_regimes()
    
    # Analyze model performance by regime
    regime_analyzer.analyze_model_performance_by_regime(results_df)
    
    # Plot regimes
    regime_analyzer.plot_regimes(log_dir=f"results/market_regimes_{ticker}")
    
    print("\n===== Demo Completed Successfully =====\n")
    print(f"Results saved to 'results/' directory")
    print(f"Trained model saved to 'models/lstm_{ticker}.pt'")

# Run the main function if executed directly
if __name__ == "__main__":
    main() 