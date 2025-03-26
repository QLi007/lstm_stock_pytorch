import torch
import yaml
import numpy as np
import argparse
import os
import logging
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from src.data.loader import StockDataLoader
from src.model.lstm import LSTMPredictor
from src.model.advanced_loss import KellyDrawdownLoss, TradingOptimizationLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def train_model(model, criterion, optimizer, dataloader, epochs, device, scheduler=None):
    """
    Train the model.
    
    Args:
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        dataloader: DataLoader with training data
        epochs: Number of epochs to train
        device: Device to use (cuda or cpu)
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        List of losses for each epoch
    """
    model.to(device)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        # Update learning rate if scheduler is provided
        if scheduler:
            scheduler.step()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return losses

def evaluate_model(model, criterion, dataloader, device):
    """
    Evaluate the model.
    
    Args:
        model: Trained model
        criterion: Loss function
        dataloader: DataLoader with evaluation data
        device: Device to use (cuda or cpu)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            
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
        'loss': total_loss / len(dataloader),
        'mean_return': portfolio_returns.mean(),
        'sharpe': portfolio_returns.mean() / (portfolio_returns.std() + 1e-6),
        'correlation': np.corrcoef(all_preds, all_targets)[0, 1]
    }
    
    return metrics

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick test without full training")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    parser.add_argument("--model-path", default="models/lstm_model.pt", help="Path to save the model")
    parser.add_argument("--loss-function", default="kelly", choices=["kelly", "trading_opt"], 
                        help="Loss function to use: kelly (KellyDrawdownLoss) or trading_opt (TradingOptimizationLoss)")
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data preparation
    loader = StockDataLoader(config_path=args.config)
    train_data, val_data, test_data = loader.prepare_data(
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1)
    )
    
    logger.info(f"Data loaded: Training={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
    
    # Create sequences
    train_sequences = loader.get_train_sequences('train')
    val_sequences = loader.get_train_sequences('val')
    
    # Convert to tensors
    X_train = torch.FloatTensor(np.array([s[0] for s in train_sequences]))
    y_train = torch.FloatTensor(np.array([s[1] for s in train_sequences]))
    
    X_val = torch.FloatTensor(np.array([s[0] for s in val_sequences]))
    y_val = torch.FloatTensor(np.array([s[1] for s in val_sequences]))
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Determine number of features
    if 'features' in config['data'] and config['data']['features']:
        num_features = len(config['data']['features'])
    else:
        # Count all numeric columns except OHLCV and target
        num_features = X_train.shape[2]  # Get from tensor shape
    
    # Initialize model
    model = LSTMPredictor(
        input_size=num_features,
        hidden_size=config['model']['hidden_size']
    )
    
    # Initialize loss function
    if args.loss_function == 'kelly':
        criterion = KellyDrawdownLoss(
            max_leverage=config['training']['max_leverage'],
            alpha=config['training'].get('kelly_alpha', 0.7),
            dd_weight=config['training'].get('drawdown_weight', 1.0),
            smoothness_weight=config['training'].get('smoothness_weight', 0.5)
        )
        logger.info("Using KellyDrawdownLoss")
    else:
        criterion = TradingOptimizationLoss(
            return_weight=config['training'].get('return_weight', 1.0),
            drawdown_weight=config['training'].get('drawdown_weight', 1.0),
            smoothness_weight=config['training'].get('smoothness_weight', 0.5),
            max_leverage=config['training']['max_leverage']
        )
        logger.info("Using TradingOptimizationLoss")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    # Check if dry-run is enabled
    if args.dry_run:
        logger.info("Running in dry-run mode with reduced epochs")
        epochs = min(5, config['training']['epochs'])
    else:
        epochs = config['training']['epochs']
    
    # Train the model
    logger.info("Starting training...")
    losses = train_model(
        model, 
        criterion, 
        optimizer, 
        train_dataloader, 
        epochs, 
        device,
        scheduler=scheduler
    )
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_model(model, criterion, val_dataloader, device)
    logger.info(f"Validation metrics: {val_metrics}")
    
    # Save model if requested
    if args.save_model:
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'val_metrics': val_metrics,
            'losses': losses
        }, args.model_path)
        
        logger.info(f"Model saved to {args.model_path}")
    
    # Print final summary
    logger.info("Training completed!")
    logger.info(f"Final validation loss: {val_metrics['loss']:.4f}")
    logger.info(f"Validation Sharpe ratio: {val_metrics['sharpe']:.4f}")
    logger.info(f"Mean return: {val_metrics['mean_return']:.4f}")

if __name__ == "__main__":
    main()
