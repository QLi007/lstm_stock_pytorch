import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Walk-forward testing framework to validate model predictions without data leakage.
    
    This class implements a point-by-point simulation of real trading, ensuring that:
    1. No future data is used in any decision
    2. Model predictions are consistent with real-time usage
    3. Performance metrics are calculated under realistic conditions
    """
    def __init__(self, model, data_loader, config, device="cpu"):
        """
        Initialize the validator.
        
        Args:
            model: Trained PyTorch model
            data_loader: Data loader instance
            config: Configuration dictionary
            device: Device to run the model on
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = device
        self.seq_length = config['training']['seq_length']
        
        # Get feature columns from config
        if 'features' in config['data'] and config['data']['features']:
            self.feature_cols = config['data']['features']
        else:
            # Default to all numeric columns except OHLCV and target
            test_data = self.data_loader.test_data
            self.feature_cols = [col for col in test_data.columns if 
                                col not in ['date', 'open', 'high', 'low', 'close', 'volume', 
                                            self.data_loader.target_col]]
        
        # Find closing price column index in features
        if 'prev_close' in self.feature_cols:
            self.close_idx = self.feature_cols.index('prev_close')
        else:
            # Default to the last feature if prev_close not found
            self.close_idx = len(self.feature_cols) - 1
            logger.warning("'prev_close' not found in features, using last feature as closing price.")
    
    def run_sequential_validation(self, data=None, log_dir="logs/sequential", plot=True):
        """
        Run walk-forward validation on data.
        
        Args:
            data: DataFrame with data. If None, uses test data from data_loader
            log_dir: Directory to save logs and plots
            plot: Whether to generate plots of the results
            
        Returns:
            DataFrame with validation results
        """
        if data is None:
            data = self.data_loader.test_data
        
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dataframe to store results
        results = pd.DataFrame()
        
        # Initialize tracking variables
        all_positions = []
        all_targets = []
        all_returns = []
        all_dates = []
        timestamps = []
        
        # Get feature data
        feature_data = data[self.feature_cols].values
        
        # Get target data
        target_data = data[self.data_loader.target_col].values
        
        # Get dates if available
        if 'date' in data.columns:
            dates = data['date'].values
        else:
            dates = np.arange(len(data))
        
        # Start at the first valid index for sequence
        start_idx = self.seq_length - 1
        
        logger.info(f"Starting sequential validation with {len(data) - start_idx} data points")
        
        # Process each data point sequentially, simulating real-time trading
        for i in tqdm(range(start_idx, len(data))):
            # Get sequence of features up to current point (avoid future data)
            features_seq = feature_data[i-self.seq_length+1:i+1]
            
            # Log current price
            current_price = feature_data[i, self.close_idx]
            
            # Get target (future return)
            true_target = target_data[i]
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_seq).unsqueeze(0).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                prediction = self.model(features_tensor).item()
            
            # Calculate return (position * future return)
            achieved_return = prediction * true_target
            
            # Record results
            all_positions.append(prediction)
            all_targets.append(true_target)
            all_returns.append(achieved_return)
            all_dates.append(dates[i])
            timestamps.append(i)
        
        # Create results dataframe
        results['date'] = all_dates
        results['timestamp'] = timestamps
        results['position'] = all_positions
        results['future_return'] = all_targets
        results['achieved_return'] = all_returns
        
        # Calculate cumulative returns
        results['cumulative_return'] = (1 + results['achieved_return']).cumprod() - 1
        
        # Calculate additional metrics
        self._calculate_metrics(results)
        
        # Log results
        self._log_results(results, log_dir)
        
        # Generate plots if requested
        if plot:
            self._generate_plots(results, log_dir)
        
        return results
    
    def _calculate_metrics(self, results):
        """Calculate performance metrics."""
        # Calculate total return
        total_return = results['cumulative_return'].iloc[-1]
        
        # Calculate Sharpe ratio (annualized)
        returns = results['achieved_return']
        sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)
        
        # Calculate maximum drawdown
        cumulative = results['cumulative_return']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / (running_max + 1)
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = (results['achieved_return'] > 0).mean()
        
        # Store metrics in results
        results['total_return'] = total_return
        results['sharpe_ratio'] = sharpe
        results['max_drawdown'] = max_drawdown
        results['win_rate'] = win_rate
        
        # Log metrics
        logger.info(f"Sequential Validation Metrics:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"  Win Rate: {win_rate:.2%}")
    
    def _log_results(self, results, log_dir):
        """Save results to CSV."""
        output_file = os.path.join(log_dir, "sequential_results.csv")
        results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    def _generate_plots(self, results, log_dir):
        """Generate and save plots."""
        # Equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results['cumulative_return'] * 100)
        plt.title('Equity Curve')
        plt.xlabel('Time Step')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "equity_curve.png"))
        
        # Position sizes over time
        plt.figure(figsize=(12, 6))
        plt.plot(results['position'])
        plt.title('Position Sizes')
        plt.xlabel('Time Step')
        plt.ylabel('Position (-1 to 1)')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "positions.png"))
        
        # Position vs Future Return scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(results['position'], results['future_return'], alpha=0.5)
        plt.title('Position vs Future Return')
        plt.xlabel('Position Size')
        plt.ylabel('Future Return')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "position_vs_return.png"))
        
        # Histogram of returns
        plt.figure(figsize=(10, 6))
        plt.hist(results['achieved_return'], bins=50, alpha=0.7)
        plt.title('Distribution of Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "return_distribution.png"))
        
        # Close all figures to free memory
        plt.close('all')


class MarketRegimeAnalyzer:
    """
    Analyzes model performance across different market regimes.
    
    Detects market regimes and evaluates model performance in each regime.
    """
    def __init__(self, results_df, lookback_period=63, volatility_threshold=0.15):
        """
        Initialize the analyzer.
        
        Args:
            results_df: DataFrame with validation results
            lookback_period: Period for regime detection (trading days)
            volatility_threshold: Threshold for high volatility regime
        """
        self.results = results_df
        self.lookback = lookback_period
        self.vol_threshold = volatility_threshold
        
        # Make sure results has future_return and position columns
        required_cols = ['future_return', 'position', 'achieved_return']
        if not all(col in self.results.columns for col in required_cols):
            raise ValueError(f"Results must contain columns: {required_cols}")
    
    def detect_regimes(self):
        """
        Detect market regimes.
        
        Returns:
            DataFrame with regime labels
        """
        # Calculate rolling returns
        rolling_returns = self.results['future_return'].rolling(self.lookback).mean()
        
        # Calculate rolling volatility
        rolling_vol = self.results['future_return'].rolling(self.lookback).std()
        
        # Initialize regime column
        self.results['regime'] = 'unknown'
        
        # Bull market: positive returns, normal volatility
        bull_mask = (rolling_returns > 0) & (rolling_vol <= self.vol_threshold)
        self.results.loc[bull_mask, 'regime'] = 'bull'
        
        # Bear market: negative returns, normal volatility
        bear_mask = (rolling_returns < 0) & (rolling_vol <= self.vol_threshold)
        self.results.loc[bear_mask, 'regime'] = 'bear'
        
        # High volatility: high volatility regardless of return direction
        vol_mask = rolling_vol > self.vol_threshold
        self.results.loc[vol_mask, 'regime'] = 'high_vol'
        
        # Sideways market: low returns (near zero), low volatility
        sideways_mask = (abs(rolling_returns) < 0.001) & (rolling_vol <= self.vol_threshold)
        self.results.loc[sideways_mask, 'regime'] = 'sideways'
        
        # Fill initial unknown values with the first known regime
        first_known = self.results['regime'].loc[self.results['regime'] != 'unknown'].iloc[0]
        self.results.loc[self.results['regime'] == 'unknown', 'regime'] = first_known
        
        return self.results
    
    def analyze_performance_by_regime(self):
        """
        Analyze model performance across different regimes.
        
        Returns:
            DataFrame with performance metrics by regime
        """
        # Make sure regimes are detected
        if 'regime' not in self.results.columns:
            self.detect_regimes()
        
        # Group by regime and calculate metrics
        regime_metrics = {}
        
        for regime in self.results['regime'].unique():
            regime_data = self.results[self.results['regime'] == regime]
            
            # Skip if not enough data
            if len(regime_data) < 5:
                continue
            
            # Calculate metrics
            returns = regime_data['achieved_return']
            
            metrics = {
                'count': len(regime_data),
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'sharpe': returns.mean() / (returns.std() + 1e-6) * np.sqrt(252),
                'win_rate': (returns > 0).mean(),
                'avg_position': regime_data['position'].abs().mean()
            }
            
            regime_metrics[regime] = metrics
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(regime_metrics).T
        metrics_df = metrics_df.sort_values('count', ascending=False)
        
        return metrics_df
    
    def plot_regime_performance(self, output_dir="logs/regimes"):
        """
        Plot performance by regime.
        
        Args:
            output_dir: Directory to save plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Make sure regimes are detected
        if 'regime' not in self.results.columns:
            self.detect_regimes()
        
        # Get metrics by regime
        metrics = self.analyze_performance_by_regime()
        
        # Plot cumulative returns by regime
        plt.figure(figsize=(12, 8))
        
        for regime in self.results['regime'].unique():
            regime_data = self.results[self.results['regime'] == regime]
            
            if len(regime_data) < 5:
                continue
                
            # Calculate cumulative returns for this regime
            cum_returns = (1 + regime_data['achieved_return']).cumprod() - 1
            
            # Plot
            plt.plot(cum_returns.values, label=f"{regime} (n={len(regime_data)})")
        
        plt.title('Cumulative Returns by Market Regime')
        plt.xlabel('Time Steps')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "returns_by_regime.png"))
        
        # Plot metrics by regime
        metrics_to_plot = ['mean_return', 'sharpe', 'win_rate']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            metrics[metric].plot(kind='bar')
            plt.title(f'{metric} by Market Regime')
            plt.ylabel(metric)
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{metric}_by_regime.png"))
        
        # Close all figures to free memory
        plt.close('all')

def run_bootstrapped_validation(validator, data, n_iterations=100, sample_size=0.8, 
                               random_seed=42, log_dir="logs/bootstrap"):
    """
    Run bootstrapped validation to assess model robustness.
    
    Args:
        validator: WalkForwardValidator instance
        data: DataFrame with data
        n_iterations: Number of bootstrap iterations
        sample_size: Size of each bootstrap sample as fraction of data
        random_seed: Random seed for reproducibility
        log_dir: Directory to save results
        
    Returns:
        DataFrame with bootstrap results
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize random number generator
    rng = np.random.RandomState(random_seed)
    
    # Initialize results storage
    bootstrap_results = {
        'total_return': [],
        'sharpe_ratio': [],
        'max_drawdown': [],
        'win_rate': []
    }
    
    logger.info(f"Running {n_iterations} bootstrap iterations with sample size {sample_size:.0%}")
    
    # Run bootstrap iterations
    for i in tqdm(range(n_iterations)):
        # Sample data with replacement
        n_samples = int(len(data) * sample_size)
        indices = rng.choice(len(data), size=n_samples, replace=True)
        sample_data = data.iloc[indices].sort_index()
        
        # Run validation on sampled data
        iteration_log_dir = os.path.join(log_dir, f"iteration_{i}")
        results = validator.run_sequential_validation(
            data=sample_data, 
            log_dir=iteration_log_dir,
            plot=(i < 5)  # Only plot first 5 iterations
        )
        
        # Store results
        bootstrap_results['total_return'].append(results['total_return'].iloc[0])
        bootstrap_results['sharpe_ratio'].append(results['sharpe_ratio'].iloc[0])
        bootstrap_results['max_drawdown'].append(results['max_drawdown'].iloc[0])
        bootstrap_results['win_rate'].append(results['win_rate'].iloc[0])
    
    # Convert to DataFrame
    results_df = pd.DataFrame(bootstrap_results)
    
    # Calculate statistics
    stats = {
        'mean': results_df.mean(),
        'std': results_df.std(),
        'min': results_df.min(),
        'max': results_df.max(),
        'median': results_df.median(),
        '5%': results_df.quantile(0.05),
        '95%': results_df.quantile(0.95)
    }
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Save results
    results_df.to_csv(os.path.join(log_dir, "bootstrap_results.csv"), index=False)
    stats_df.to_csv(os.path.join(log_dir, "bootstrap_stats.csv"))
    
    # Plot histograms
    for col in bootstrap_results.keys():
        plt.figure(figsize=(10, 6))
        plt.hist(results_df[col], bins=20, alpha=0.7)
        plt.axvline(stats['mean'][col], color='r', linestyle='--', label=f"Mean: {stats['mean'][col]:.4f}")
        plt.axvline(stats['5%'][col], color='g', linestyle=':', label=f"5% CI: {stats['5%'][col]:.4f}")
        plt.axvline(stats['95%'][col], color='g', linestyle=':', label=f"95% CI: {stats['95%'][col]:.4f}")
        plt.title(f'Bootstrap Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f"{col}_distribution.png"))
    
    # Close all figures to free memory
    plt.close('all')
    
    logger.info(f"Bootstrap results saved to {log_dir}")
    logger.info("\nBootstrap Statistics:")
    for metric in stats_df.columns:
        logger.info(f"{metric}:")
        logger.info(f"  Mean: {stats['mean'][metric]:.4f}")
        logger.info(f"  Std: {stats['std'][metric]:.4f}")
        logger.info(f"  5-95% CI: [{stats['5%'][metric]:.4f}, {stats['95%'][metric]:.4f}]")
    
    return results_df, stats_df 