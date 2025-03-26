import torch
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data.loader import StockDataLoader
from src.model.lstm import LSTMPredictor
from src.baselines.simple_strategies import SimpleMovingAverageCrossover, MomentumStrategy

def calculate_metrics(predictions, targets, full_window=True):
    """Calculate comprehensive evaluation metrics"""
    # Convert tensors to numpy arrays if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()
    
    # Basic prediction accuracy metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    
    # Calculate directional accuracy (sign agreement)
    correct_direction = np.sum(np.sign(predictions) == np.sign(targets))
    direction_accuracy = correct_direction / len(targets)
    
    # Portfolio returns
    portfolio_returns = predictions * targets  # Position * Return
    
    # Risk-adjusted return metrics
    annualization_factor = np.sqrt(252)  # For daily data
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns) + 1e-6
    
    # Sharpe ratio (annualized)
    sharpe = mean_return / std_return * annualization_factor
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (peak + 1)  # Relative drawdown
    max_drawdown = np.max(drawdown)
    
    # Calmar Ratio (annualized return / maximum drawdown)
    if max_drawdown > 0:
        calmar = (mean_return * 252) / max_drawdown
    else:
        calmar = float('inf')
    
    # Sortino Ratio (using only negative returns for downside risk)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = np.std(downside_returns) + 1e-6
        sortino = mean_return / downside_deviation * annualization_factor
    else:
        sortino = float('inf')
    
    # Win Rate
    winning_trades = np.sum(portfolio_returns > 0)
    total_trades = len(portfolio_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Profit/Loss Ratio
    avg_profit = np.mean(portfolio_returns[portfolio_returns > 0]) if np.any(portfolio_returns > 0) else 0
    avg_loss = np.abs(np.mean(portfolio_returns[portfolio_returns < 0])) if np.any(portfolio_returns < 0) else 1e-6
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
    
    # Results dictionary
    metrics = {
        # Prediction accuracy metrics
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        
        # Risk-adjusted return metrics
        'mean_return': mean_return,
        'annualized_return': mean_return * 252,
        'std_return': std_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        
        # Trading performance metrics
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'final_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    }
    
    return metrics

def statistical_significance_test(strategy_returns, benchmark_returns):
    """
    Perform statistical tests to evaluate if strategy outperforms benchmark.
    
    Args:
        strategy_returns: Returns from the strategy
        benchmark_returns: Returns from the benchmark (e.g., buy and hold)
        
    Returns:
        Dictionary with test results
    """
    # t-test for mean difference
    t_stat, p_value_t = stats.ttest_ind(strategy_returns, benchmark_returns, equal_var=False)
    
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, p_value_w = stats.wilcoxon(strategy_returns, benchmark_returns)
    
    # Bootstrap analysis
    n_bootstrap = 1000
    bootstrap_differences = []
    
    for _ in range(n_bootstrap):
        # Random resampling with replacement
        strategy_sample = np.random.choice(strategy_returns, size=len(strategy_returns), replace=True)
        benchmark_sample = np.random.choice(benchmark_returns, size=len(benchmark_returns), replace=True)
        
        bootstrap_differences.append(np.mean(strategy_sample) - np.mean(benchmark_sample))
    
    # Confidence interval from bootstrap
    confidence_interval = np.percentile(bootstrap_differences, [2.5, 97.5])
    
    return {
        't_statistic': t_stat,
        'p_value_t': p_value_t,
        'w_statistic': w_stat,
        'p_value_w': p_value_w,
        'bootstrap_mean_diff': np.mean(bootstrap_differences),
        'bootstrap_95ci_lower': confidence_interval[0],
        'bootstrap_95ci_upper': confidence_interval[1],
        'outperforms_95ci': confidence_interval[0] > 0
    }

def visualize_predictions(predictions, targets, save_path=None):
    """Visualize model predictions vs actual values with enhanced plots"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()
    
    # Calculate portfolio returns
    portfolio_returns = predictions * targets
    
    # Use portfolio returns to calculate multiple performance metrics
    metrics = calculate_metrics(predictions, targets)
    
    # Create subplots for visualization
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot 1: Predictions vs Actual Returns
    axs[0].plot(targets, label='Actual Returns', alpha=0.7)
    axs[0].plot(predictions, label='Predicted Position Size', alpha=0.7)
    axs[0].set_title('Predictions vs Actual Returns')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Returns Comparison
    # Calculate cumulative returns for model strategy and buy & hold
    cumulative_portfolio = np.cumprod(1 + portfolio_returns) - 1
    cumulative_buyhold = np.cumprod(1 + targets) - 1
    
    # Create random portfolios for comparison (Monte Carlo)
    n_random = 20
    random_portfolios = []
    for _ in range(n_random):
        random_positions = np.random.uniform(-1, 1, size=len(targets))
        random_returns = random_positions * targets
        random_portfolio = np.cumprod(1 + random_returns) - 1
        random_portfolios.append(random_portfolio)
    
    # Plot random portfolios in background
    for i, portfolio in enumerate(random_portfolios):
        axs[1].plot(portfolio, color='gray', alpha=0.2, linewidth=0.5)
    
    # Plot main strategies
    axs[1].plot(cumulative_portfolio, label=f'Model Strategy (Sharpe: {metrics["sharpe"]/np.sqrt(252):.2f})', 
                color='blue', linewidth=2)
    axs[1].plot(cumulative_buyhold, label='Buy & Hold', color='green', linewidth=2)
    
    # Highlight drawdowns
    peak = np.maximum.accumulate(cumulative_portfolio)
    drawdowns = (peak - cumulative_portfolio)
    max_dd_idx = np.argmax(drawdowns)
    if max_dd_idx > 0:
        max_dd_start = np.where(peak[max_dd_idx] == cumulative_portfolio)[0][-1]
        axs[1].fill_between(range(max_dd_start, max_dd_idx+1), 
                           cumulative_portfolio[max_dd_start:max_dd_idx+1], 
                           peak[max_dd_start:max_dd_idx+1], 
                           color='red', alpha=0.3)
        axs[1].text(max_dd_idx, peak[max_dd_idx], f"Max DD: {metrics['max_drawdown']:.2%}", 
                   verticalalignment='bottom', horizontalalignment='center')
    
    axs[1].set_title('Cumulative Returns Comparison with Maximum Drawdown Highlighted')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling Performance Metrics
    window = min(60, len(portfolio_returns) // 4)  # Dynamic window size
    
    if len(portfolio_returns) > window:
        # Calculate rolling metrics
        rolling_returns = pd.Series(portfolio_returns)
        rolling_sharpe = rolling_returns.rolling(window).mean() / rolling_returns.rolling(window).std() * np.sqrt(252)
        rolling_vol = rolling_returns.rolling(window).std() * np.sqrt(252)
        
        # Plot rolling metrics
        ax_sharpe = axs[2]
        ax_vol = ax_sharpe.twinx()
        
        ax_sharpe.plot(rolling_sharpe, label='Rolling Sharpe (60d)', color='blue')
        ax_vol.plot(rolling_vol, label='Rolling Volatility (60d)', color='red', linestyle='--')
        
        ax_sharpe.set_title('Rolling Performance Metrics')
        ax_sharpe.set_ylabel('Sharpe Ratio', color='blue')
        ax_vol.set_ylabel('Annualized Volatility', color='red')
        
        # Add separate legends for each y-axis
        lines_1, labels_1 = ax_sharpe.get_legend_handles_labels()
        lines_2, labels_2 = ax_vol.get_legend_handles_labels()
        ax_sharpe.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
        ax_sharpe.grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "Insufficient data for rolling metrics", 
                   ha='center', va='center', transform=axs[2].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Add a text box with key performance metrics
    metrics_text = (
        f"Sharpe Ratio: {metrics['sharpe']/np.sqrt(252):.2f}\n"
        f"Sortino Ratio: {metrics['sortino']/np.sqrt(252):.2f}\n"
        f"Calmar Ratio: {metrics['calmar']/np.sqrt(252):.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Profit/Loss Ratio: {metrics['profit_loss_ratio']:.2f}\n"
        f"Final Return: {metrics['final_return']:.2%}\n"
        f"Direction Accuracy: {metrics['direction_accuracy']:.2%}"
    )
    
    # Add a text box at top right of figure
    plt.figtext(0.95, 0.95, metrics_text, ha='right', va='top', 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.show()
    
    return metrics

def compare_with_baselines(X, y, model, data_dates=None):
    """
    Compare the LSTM model with baseline strategies
    
    Args:
        X: Feature matrix
        y: Target returns
        model: Trained LSTM model
        data_dates: Dates corresponding to the data points (if available)
    
    Returns:
        Dictionary with comparison results
    """
    # Get model predictions
    if isinstance(X, torch.Tensor):
        model_predictions = model(X).squeeze().detach().numpy()
    else:
        model_predictions = model(torch.FloatTensor(X)).squeeze().detach().numpy()
    
    if isinstance(y, torch.Tensor):
        actual_returns = y.numpy()
    else:
        actual_returns = y
    
    # Calculate model performance
    model_performance = calculate_metrics(model_predictions, actual_returns)
    model_returns = model_predictions * actual_returns
    
    # Buy & Hold strategy (baseline 1)
    buyhold_performance = calculate_metrics(np.ones_like(actual_returns), actual_returns)
    buyhold_returns = actual_returns  # Position of 1.0 * returns
    
    # SMA Crossover strategy (baseline 2)
    # Assuming X contains price data in the last feature
    if X.shape[2] >= 1:  # Check if we have enough features
        # For this example, let's assume the close price is the last feature
        close_prices = X[:, -1, -1]  # Last feature, last time step
        
        sma_strategy = SimpleMovingAverageCrossover(short_window=5, long_window=20)
        sma_positions = sma_strategy.generate_positions(close_prices)
        sma_performance = calculate_metrics(sma_positions, actual_returns)
        sma_returns = sma_positions * actual_returns
        
        # Momentum strategy (baseline 3)
        momentum_strategy = MomentumStrategy(window=10)
        momentum_positions = momentum_strategy.generate_positions(close_prices)
        momentum_performance = calculate_metrics(momentum_positions, actual_returns)
        momentum_returns = momentum_positions * actual_returns
    else:
        sma_performance = {"error": "Not enough features for SMA strategy"}
        momentum_performance = {"error": "Not enough features for Momentum strategy"}
        sma_returns = np.zeros_like(actual_returns)
        momentum_returns = np.zeros_like(actual_returns)
    
    # Statistical significance tests
    model_vs_buyhold = statistical_significance_test(model_returns, buyhold_returns)
    model_vs_sma = statistical_significance_test(model_returns, sma_returns)
    model_vs_momentum = statistical_significance_test(model_returns, momentum_returns)
    
    # Create visualization
    strategies = {
        "LSTM Model": model_returns,
        "Buy & Hold": buyhold_returns,
        "SMA Crossover": sma_returns,
        "Momentum": momentum_returns
    }
    
    # Calculate cumulative returns for each strategy
    cum_returns = {}
    for name, returns in strategies.items():
        cum_returns[name] = np.cumprod(1 + returns) - 1
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    for name, returns in cum_returns.items():
        plt.plot(returns, label=name)
    
    plt.title('Strategy Comparison: Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add date labels if available
    if data_dates is not None and len(data_dates) == len(actual_returns):
        # Show only a subset of dates to avoid overcrowding
        n_ticks = min(10, len(data_dates))
        tick_indices = np.linspace(0, len(data_dates)-1, n_ticks, dtype=int)
        plt.xticks(tick_indices, [data_dates[i] for i in tick_indices], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Return comparison results
    return {
        "model_performance": model_performance,
        "buyhold_performance": buyhold_performance,
        "sma_performance": sma_performance,
        "momentum_performance": momentum_performance,
        "statistical_tests": {
            "model_vs_buyhold": model_vs_buyhold,
            "model_vs_sma": model_vs_sma,
            "model_vs_momentum": model_vs_momentum
        }
    }

def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance for the LSTM model using a permutation approach
    
    Args:
        model: Trained LSTM model
        feature_names: List of feature names corresponding to input dimensions
        
    Returns:
        DataFrame with feature importance scores
    """
    # Get model parameters
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'lstm' in name and 'ih' in name:
            # Get the input weights from LSTM layer
            weights.append(param.detach().cpu().numpy())
    
    if not weights:
        return pd.DataFrame({"error": ["Could not extract feature weights from model"]})
    
    # For simplicity, use the first LSTM layer's weights
    feature_weights = np.abs(weights[0])
    
    # Average across all hidden units to get importance per feature
    importance_scores = feature_weights.mean(axis=0)
    
    # Create a DataFrame for better visualization
    if len(feature_names) != len(importance_scores):
        # If mismatch, create generic feature names
        feature_names = [f"Feature_{i}" for i in range(len(importance_scores))]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importance_scores)],
        'Importance': importance_scores
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('LSTM Feature Importance Analysis')
    plt.gca().invert_yaxis()  # Display highest importance at the top
    plt.tight_layout()
    plt.show()
    
    return importance_df

def main():
    parser = argparse.ArgumentParser(description="Advanced evaluation of trained LSTM model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model-path", default="models/lstm_model.pt", help="Path to trained model")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--compare-baselines", action="store_true", help="Compare with baseline strategies")
    parser.add_argument("--analyze-features", action="store_true", help="Analyze feature importance")
    parser.add_argument("--save-plot", default=None, help="Path to save visualization plot")
    parser.add_argument("--loss-function", default="mse", help="Loss function used for training")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load test data
    loader = StockDataLoader(config_path=args.config)
    
    # Get data split
    train_data, val_data, test_data = loader.prepare_data(
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1)
    )
    
    # Get test sequences
    test_sequences = loader.get_train_sequences('test')
    
    # Convert to Tensors
    X = torch.FloatTensor(np.array([s[0] for s in test_sequences]))
    y = torch.FloatTensor(np.array([s[1] for s in test_sequences]))
    
    # Get dates if available
    data_dates = None
    if hasattr(loader, 'test_dates'):
        data_dates = loader.test_dates
    
    # Initialize model
    input_size = X.shape[2] if X.shape[2] > 0 else config['model']['input_size']
    model = LSTMPredictor(
        input_size=input_size, 
        hidden_size=config['model']['hidden_size']
    )
    
    # Load trained model weights
    checkpoint = torch.load(args.model_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Get feature names if available
    feature_names = config['data'].get('features', [f"Feature_{i}" for i in range(input_size)])
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X).squeeze()
    
    # Evaluate model with enhanced metrics
    metrics = calculate_metrics(predictions, y)
    
    # Print detailed metrics
    print("\n" + "="*50)
    print("ADVANCED MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Model Path: {args.model_path}")
    print(f"Loss Function: {args.loss_function}")
    print("="*50)
    
    # Prediction accuracy metrics
    print("\nPrediction Accuracy Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
    
    # Risk-adjusted return metrics
    print("\nRisk-Adjusted Return Metrics:")
    print(f"Mean Daily Return: {metrics['mean_return']:.4f}")
    print(f"Annualized Return: {metrics['annualized_return']:.4f}")
    print(f"Annualized Volatility: {metrics['std_return'] * np.sqrt(252):.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
    print(f"Sortino Ratio: {metrics['sortino']:.4f}")
    print(f"Calmar Ratio: {metrics['calmar']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.4f}")
    
    # Trading performance metrics
    print("\nTrading Performance Metrics:")
    print(f"Win Rate: {metrics['win_rate']:.4f}")
    print(f"Profit/Loss Ratio: {metrics['profit_loss_ratio']:.4f}")
    print(f"Final Return: {metrics['final_return']:.4f}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating enhanced visualization plots...")
        visualize_predictions(predictions, y, args.save_plot)
    
    # Compare with baseline strategies
    if args.compare_baselines:
        print("\nComparing model with baseline strategies...")
        comparison = compare_with_baselines(X, y, model, data_dates)
        
        # Print comparative statistics
        print("\nComparative Statistics:")
        print(f"LSTM Sharpe Ratio: {comparison['model_performance']['sharpe']:.4f}")
        print(f"Buy & Hold Sharpe Ratio: {comparison['buyhold_performance']['sharpe']:.4f}")
        
        if isinstance(comparison['sma_performance'], dict) and 'sharpe' in comparison['sma_performance']:
            print(f"SMA Crossover Sharpe Ratio: {comparison['sma_performance']['sharpe']:.4f}")
        
        if isinstance(comparison['momentum_performance'], dict) and 'sharpe' in comparison['momentum_performance']:
            print(f"Momentum Sharpe Ratio: {comparison['momentum_performance']['sharpe']:.4f}")
        
        # Print statistical significance test results
        test_results = comparison['statistical_tests']['model_vs_buyhold']
        print("\nStatistical Tests (LSTM vs Buy & Hold):")
        print(f"t-test p-value: {test_results['p_value_t']:.4f} {'*' if test_results['p_value_t'] < 0.05 else ''}")
        print(f"Bootstrap 95% CI: [{test_results['bootstrap_95ci_lower']:.4f}, {test_results['bootstrap_95ci_upper']:.4f}]")
        print(f"Statistically Outperforms at 95% CI: {'Yes' if test_results['outperforms_95ci'] else 'No'}")
    
    # Analyze feature importance
    if args.analyze_features:
        print("\nAnalyzing feature importance...")
        importance_df = analyze_feature_importance(model, feature_names)
        
        # Print top 10 most important features
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

if __name__ == "__main__":
    main() 