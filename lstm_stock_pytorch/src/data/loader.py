import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from sklearn.model_selection import train_test_split
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class StockDataLoader:
    def __init__(self, config_path='configs/default.yaml'):
        """
        Initialize the stock data loader.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = Path(self.config['data']['path'])
        self.seq_length = self.config['training']['seq_length']
        self.target_col = self.config['data'].get('target', 'future_5d_return')
        self.target_offset = int(self.config['data'].get('target_offset', 5))
        self.normalize_method = self.config['data'].get('normalize', 'standard')
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(
            price_col='close',
            normalize_method=self.normalize_method,
            feature_window=self.config['data'].get('feature_window', 250),
            scale_separately=self.config['data'].get('scale_separately', True)
        )
        
        # Flag to track if data is prepared
        self.data_prepared = False
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def load_data(self):
        """
        Load and preprocess the stock data.
        
        Returns:
            Processed DataFrame with features and target
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Read data
        try:
            df = pd.read_csv(self.data_path, parse_dates=['date'])
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate features
        df = self.feature_engineer.calculate_features(df)
        
        # Add target (future return)
        df[f'future_{self.target_offset}d_return'] = df['close'].pct_change(self.target_offset).shift(-self.target_offset)
        
        # Normalize features
        feature_cols = self.config['data'].get('features', None)
        if feature_cols is None:
            # Get all numeric columns except date, OHLCV, and target
            feature_cols = [col for col in df.columns if 
                           col not in ['date', 'open', 'high', 'low', 'close', 'volume', self.target_col]]
        
        # Normalize features if specified
        if self.normalize_method:
            df = self.feature_engineer.normalize_features(df, feature_cols)
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Ensure no future data leakage
        self._check_future_leakage(df)
        
        return df
    
    def prepare_data(self, test_size=0.2, val_size=0.1, shuffle=False):
        """
        Prepare data splits for training, validation, and testing.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            shuffle: Whether to shuffle the data (not recommended for time series)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.data_prepared:
            return self.train_data, self.val_data, self.test_data
        
        df = self.load_data()
        
        if not shuffle:
            # Time-based split (recommended for time series)
            test_idx = int(len(df) * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))
            
            train_data = df.iloc[:val_idx]
            val_data = df.iloc[val_idx:test_idx]
            test_data = df.iloc[test_idx:]
        else:
            # Random split (not recommended for time series)
            train_val_data, test_data = train_test_split(df, test_size=test_size, shuffle=shuffle)
            train_data, val_data = train_test_split(train_val_data, test_size=val_size/(1-test_size), shuffle=shuffle)
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.data_prepared = True
        
        logger.info(f"Data prepared: Training={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def get_train_sequences(self, split='train'):
        """
        Get sequence data for training.
        
        Args:
            split: Data split to use ('train', 'val', or 'test')
            
        Returns:
            Array of (features, target) sequences
        """
        # Prepare data if not already done
        if not self.data_prepared:
            self.prepare_data()
        
        # Select the appropriate split
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        elif split == 'test':
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Extract features and target
        feature_cols = self.config['data'].get('features', None)
        if feature_cols is None:
            # Use all numeric columns except date, OHLCV, and target
            feature_cols = [col for col in data.columns if 
                           col not in ['date', 'open', 'high', 'low', 'close', 'volume', self.target_col]]
        
        features = data[feature_cols].values
        targets = data[self.target_col].values
        
        # Create sequences
        sequences = []
        for i in range(len(features) - self.seq_length + 1):
            # Extract sequence of features
            seq_features = features[i:i+self.seq_length]
            
            # Target is the value at the end of the sequence
            seq_target = targets[i+self.seq_length-1]
            
            sequences.append((seq_features, seq_target))
        
        return np.array(sequences, dtype=object)
    
    def _check_future_leakage(self, df):
        """Check for potential future data leakage."""
        # Calculate correlation between features and future returns
        future_return = df['close'].shift(-1) / df['close'] - 1
        
        suspicious_cols = []
        for col in df.columns:
            if col not in ['date', 'open', 'high', 'low', 'close', 'volume', self.target_col]:
                corr = df[col].corr(future_return)
                if abs(corr) > 0.8:  # Arbitrary threshold
                    suspicious_cols.append((col, corr))
        
        if suspicious_cols:
            logger.warning("Potential future data leakage detected:")
            for col, corr in suspicious_cols:
                logger.warning(f"  {col}: correlation={corr:.4f}")
        
        return len(suspicious_cols) == 0
