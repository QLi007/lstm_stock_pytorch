import torch
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data.loader import StockDataLoader
from src.model.lstm import LSTMPredictor

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    # Convert tensors to numpy arrays if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    
    # Calculate directional accuracy (sign agreement)
    correct_direction = np.sum(np.sign(predictions) == np.sign(targets))
    direction_accuracy = correct_direction / len(targets)
    
    # Sharpe ratio (simplified)
    returns = predictions * targets  # Portfolio returns
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
    
    return {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'sharpe': sharpe
    }

def visualize_predictions(predictions, targets, save_path=None):
    """Visualize model predictions vs actual values"""
    plt.figure(figsize=(12, 6))
    
    # Plot targets and predictions
    plt.subplot(2, 1, 1)
    plt.plot(targets, label='Actual Returns', alpha=0.7)
    plt.plot(predictions, label='Predicted Position Size', alpha=0.7)
    plt.legend()
    plt.title('Predictions vs Actual Returns')
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    portfolio_returns = predictions * targets
    cumulative_returns = np.cumsum(portfolio_returns)
    buy_hold_returns = np.cumsum(targets)
    
    plt.plot(cumulative_returns, label='Model Strategy', alpha=0.7)
    plt.plot(buy_hold_returns, label='Buy & Hold', alpha=0.7)
    plt.legend()
    plt.title('Cumulative Returns')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model-path", default="models/lstm_model.pt", help="Path to trained model")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--save-plot", default=None, help="Path to save visualization plot")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load test data
    loader = StockDataLoader(config_path=args.config)
    sequences = loader.get_train_data()  # In a real scenario, this would be test data
    
    # Convert to Tensors
    X = torch.FloatTensor(np.array([s[0] for s in sequences]))
    y = torch.FloatTensor(np.array([s[1] for s in sequences]))
    
    # Initialize model
    model = LSTMPredictor(
        input_size=config['model']['input_size'], 
        hidden_size=config['model']['hidden_size']
    )
    
    # Load trained model weights
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X).squeeze()
    
    # Evaluate model
    metrics = calculate_metrics(predictions, y)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_predictions(predictions, y, args.save_plot)

if __name__ == "__main__":
    main() 