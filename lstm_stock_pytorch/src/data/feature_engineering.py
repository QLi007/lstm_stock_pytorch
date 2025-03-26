import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for stock data."""
    
    def __init__(self, price_col='close', normalize_method='standard', 
                 feature_window=250, scale_separately=True):
        """
        Initialize feature engineering.
        
        Args:
            price_col: Column name for the price data
            normalize_method: 'standard', 'minmax', or None
            feature_window: Window size for normalization (e.g., 250 trading days)
            scale_separately: Whether to scale each feature separately
        """
        self.price_col = price_col
        self.normalize_method = normalize_method
        self.feature_window = feature_window
        self.scale_separately = scale_separately
        self.scalers = {}  # Store scalers for each feature
        
    def calculate_features(self, df):
        """
        Calculate all technical indicators for the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Make sure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # Essential lag features - using only past data
        data['prev_close'] = data[self.price_col].shift(1)
        data['return_1d'] = data[self.price_col].pct_change(1)
        
        # Moving Averages (multiple periods)
        for period in [5, 10, 20, 34, 60, 120, 200]:
            data[f'MA_{period}'] = data['prev_close'].rolling(window=period).mean()
            # Also add relative position to MA
            data[f'price_to_MA_{period}'] = data['prev_close'] / data[f'MA_{period}'] - 1
        
        # Volatility features
        for period in [10, 20, 60]:
            data[f'volatility_{period}'] = data['return_1d'].rolling(window=period).std()
        
        # RSI calculation
        for period in [6, 14, 28]:
            data[f'RSI_{period}'] = self._calculate_rsi(data['prev_close'], period)
        
        # MACD calculation with different parameters
        data['MACD'] = self._calculate_macd(data['prev_close'])
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        # Williams %R indicator
        for period in [14, 28]:
            data[f'Williams_%R_{period}'] = self._calculate_williams_r(
                data['prev_close'], data['high'], data['low'], period)
        
        # Volume indicators
        if 'volume' in data.columns:
            data['volume_ma10'] = data['volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma10']
        
        # Ensure we don't use future data by shifting target
        # No need to shift features as they're all calculated from 'prev_close' already
        
        # Sanity check: all features should be based on past data only
        for col in data.columns:
            if col not in ['date', 'open', 'high', 'low', 'close', 'volume']:
                # Ensure no future leakage by checking correlation with future returns
                if 'date' in data.columns:
                    future_corr = data[col].corr(data['close'].shift(-1) / data['close'] - 1)
                    if abs(future_corr) > 0.9:  # Arbitrary threshold for suspiciously high correlation
                        logger.warning(f"Feature {col} has suspiciously high correlation with future returns: {future_corr}")
        
        return data
    
    def normalize_features(self, df, feature_columns=None):
        """
        Normalize features using the specified method.
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to normalize, if None all numeric columns will be used
            
        Returns:
            DataFrame with normalized features
        """
        if self.normalize_method is None:
            return df
        
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # If no feature columns specified, use all numeric columns except date and target
        if feature_columns is None:
            feature_columns = [col for col in data.columns if 
                              col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Initialize scaler based on the specified method
        if self.normalize_method == 'standard':
            scaler_class = StandardScaler
        elif self.normalize_method == 'minmax':
            scaler_class = MinMaxScaler
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
        
        if self.scale_separately:
            # Scale each feature separately
            for col in feature_columns:
                if col not in self.scalers:
                    self.scalers[col] = scaler_class()
                
                # Use a rolling window approach to avoid look-ahead bias
                for i in range(self.feature_window, len(data)):
                    # Fit on historical data only
                    window_data = data.iloc[i-self.feature_window:i][col].values.reshape(-1, 1)
                    self.scalers[col].fit(window_data)
                    
                    # Transform only the current value
                    data.loc[data.index[i], col] = self.scalers[col].transform(
                        [[data.iloc[i][col]]])[0][0]
        else:
            # Scale all features together
            if 'all_features' not in self.scalers:
                self.scalers['all_features'] = scaler_class()
            
            # Use a rolling window approach to avoid look-ahead bias
            for i in range(self.feature_window, len(data)):
                # Fit on historical data only
                window_data = data.iloc[i-self.feature_window:i][feature_columns].values
                self.scalers['all_features'].fit(window_data)
                
                # Transform only the current row
                data.loc[data.index[i], feature_columns] = self.scalers['all_features'].transform(
                    [data.iloc[i][feature_columns].values])[0]
        
        return data
    
    def _calculate_rsi(self, series, period=14):
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-9)  # Add small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        """Calculate Moving Average Convergence Divergence."""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd
    
    def _calculate_williams_r(self, close, high, low, period=14):
        """Calculate Williams %R indicator."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
        return wr 