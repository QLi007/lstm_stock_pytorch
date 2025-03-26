import numpy as np
import pandas as pd

class SimpleMovingAverageCrossover:
    """
    Simple Moving Average Crossover trading strategy.
    
    Buy signal: Short-term MA crosses above long-term MA
    Sell signal: Short-term MA crosses below long-term MA
    """
    def __init__(self, short_window=5, long_window=20, max_leverage=1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.max_leverage = max_leverage
        
    def generate_positions(self, prices):
        """
        Generate trading positions based on SMA crossover.
        
        Args:
            prices: Array of price data
            
        Returns:
            Array of position sizes (-1 to 1)
        """
        if len(prices) < self.long_window:
            raise ValueError(f"Price data too short for SMA strategy. Need at least {self.long_window} points.")
        
        # Calculate short and long-term moving averages
        short_ma = np.convolve(prices, np.ones(self.short_window)/self.short_window, mode='valid')
        long_ma = np.convolve(prices, np.ones(self.long_window)/self.long_window, mode='valid')
        
        # Align arrays (valid convolution shortens the array)
        short_ma_aligned = short_ma[-(len(long_ma)):]
        
        # Generate signals based on crossovers
        signals = np.zeros(len(prices))
        
        # Calculate signal based on MA relationship
        positions = np.zeros(len(prices))
        
        # Fill in positions based on crossover strategy
        for i in range(len(long_ma)):
            idx = i + (len(prices) - len(long_ma))
            if short_ma_aligned[i] > long_ma[i]:
                positions[idx] = self.max_leverage  # Long position
            elif short_ma_aligned[i] < long_ma[i]:
                positions[idx] = -self.max_leverage  # Short position
        
        # Fill initial positions based on early signals
        if len(long_ma) > 0:
            positions[:len(prices) - len(long_ma)] = positions[len(prices) - len(long_ma)]
        
        return positions


class MomentumStrategy:
    """
    Momentum trading strategy.
    
    Buy signal: Price has increased over the window period
    Sell signal: Price has decreased over the window period
    """
    def __init__(self, window=10, max_leverage=1.0, threshold=0.0):
        self.window = window
        self.max_leverage = max_leverage
        self.threshold = threshold
        
    def generate_positions(self, prices):
        """
        Generate trading positions based on momentum.
        
        Args:
            prices: Array of price data
            
        Returns:
            Array of position sizes (-1 to 1)
        """
        if len(prices) < self.window:
            raise ValueError(f"Price data too short for momentum strategy. Need at least {self.window} points.")
        
        # Calculate returns over the momentum window
        returns = np.zeros(len(prices))
        
        for i in range(self.window, len(prices)):
            momentum_return = (prices[i] / prices[i - self.window]) - 1
            
            # Apply threshold filter
            if momentum_return > self.threshold:
                returns[i] = self.max_leverage  # Long position
            elif momentum_return < -self.threshold:
                returns[i] = -self.max_leverage  # Short position
            else:
                returns[i] = 0  # No position
        
        # Fill initial positions based on first valid signal
        first_valid = next((i for i, r in enumerate(returns) if r != 0), None)
        if first_valid is not None:
            returns[:first_valid] = returns[first_valid]
        
        return returns


class MeanReversionStrategy:
    """
    Mean Reversion trading strategy based on Bollinger Bands.
    
    Buy signal: Price falls below lower band
    Sell signal: Price rises above upper band
    """
    def __init__(self, window=20, num_std=2.0, max_leverage=1.0):
        self.window = window
        self.num_std = num_std
        self.max_leverage = max_leverage
        
    def generate_positions(self, prices):
        """
        Generate trading positions based on mean reversion.
        
        Args:
            prices: Array of price data
            
        Returns:
            Array of position sizes (-1 to 1)
        """
        if len(prices) < self.window:
            raise ValueError(f"Price data too short for mean reversion strategy. Need at least {self.window} points.")
        
        # Calculate rolling mean and standard deviation
        positions = np.zeros(len(prices))
        
        for i in range(self.window, len(prices)):
            window_slice = prices[i-self.window:i]
            mean = np.mean(window_slice)
            std = np.std(window_slice)
            
            upper_band = mean + (self.num_std * std)
            lower_band = mean - (self.num_std * std)
            
            current_price = prices[i]
            
            if current_price < lower_band:
                positions[i] = self.max_leverage  # Long position - buy when price is low
            elif current_price > upper_band:
                positions[i] = -self.max_leverage  # Short position - sell when price is high
            else:
                # Optional: scale position based on distance from mean
                distance_from_mean = (current_price - mean) / (upper_band - mean)
                positions[i] = -distance_from_mean * self.max_leverage
        
        # Fill initial positions
        positions[:self.window] = 0
        
        return positions


class EnsembleStrategy:
    """
    Ensemble strategy that combines multiple strategies' signals.
    
    Weights can be adjusted for each strategy.
    """
    def __init__(self, strategies, weights=None, max_leverage=1.0):
        """
        Initialize ensemble strategy.
        
        Args:
            strategies: List of strategy objects
            weights: List of weights for each strategy (if None, equal weighting is used)
            max_leverage: Maximum leverage allowed
        """
        self.strategies = strategies
        self.max_leverage = max_leverage
        
        if weights is None:
            self.weights = [1/len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("Number of weights must match number of strategies")
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w/total for w in weights]
    
    def generate_positions(self, prices):
        """
        Generate trading positions based on ensemble of strategies.
        
        Args:
            prices: Array of price data
            
        Returns:
            Array of position sizes (-1 to 1)
        """
        # Get positions from each strategy
        all_positions = []
        for strategy in self.strategies:
            try:
                strategy_positions = strategy.generate_positions(prices)
                all_positions.append(strategy_positions)
            except Exception as e:
                print(f"Strategy {strategy.__class__.__name__} failed: {str(e)}")
                # Add zeros if strategy fails
                all_positions.append(np.zeros(len(prices)))
        
        # Combine positions using weights
        ensemble_positions = np.zeros(len(prices))
        for i, positions in enumerate(all_positions):
            ensemble_positions += positions * self.weights[i]
        
        # Limit to max leverage
        ensemble_positions = np.clip(ensemble_positions, -self.max_leverage, self.max_leverage)
        
        return ensemble_positions 