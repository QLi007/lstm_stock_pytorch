import json
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def generate_notebook():
    """
    Generate an improved Colab notebook for LSTM stock prediction with sequential validation.
    """
    # Create a new notebook
    nb = new_notebook()
    
    # Add title and introduction
    nb.cells.append(new_markdown_cell("""
    # LSTM Stock Prediction with Robust Sequential Validation
    
    **Features:**
    * Clone repository and set up environment
    * Download and preprocess stock data from Yahoo Finance
    * Run LSTM model training with advanced loss functions
    * Compare with baseline strategies
    * Sequential walk-forward backtesting (without data leakage)
    * Market regime analysis and performance visualization
    * Bootstrapped validation for robustness testing
    """))
    
    # Setup section
    nb.cells.append(new_markdown_cell("## Environment Setup"))
    
    # Clone repository
    nb.cells.append(new_code_cell("""
    # Install required packages
    !pip install yfinance pandas-ta scikit-learn scipy tensorflow matplotlib seaborn tqdm
    
    # Clone the repository
    !git clone https://github.com/QLi007/lstm_stock_pytorch.git
    %cd lstm_stock_pytorch
    
    # Install requirements
    !pip install -r requirements.txt
    """))
    
    # Data download section
    nb.cells.append(new_markdown_cell("## Download Stock Data from Yahoo Finance"))
    nb.cells.append(new_code_cell("""
    import os
    import yfinance as yf
    import pandas as pd
    import pandas_ta as ta
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define stock tickers to download (you can change these)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Set date range for 5 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Download data for each ticker
    for ticker in tickers:
        print(f"Downloading {ticker} data...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Add technical indicators
        # Moving averages
        for period in [5, 10, 20, 34, 60, 120, 200]:
            data[f'MA_{period}'] = data['close'].rolling(window=period).mean()
            data[f'price_to_MA_{period}'] = data['close'] / data[f'MA_{period}'] - 1
        
        # RSI
        for period in [6, 14, 28]:
            data[f'RSI_{period}'] = ta.rsi(data['close'], length=period)
        
        # MACD
        macd = ta.macd(data['close'])
        data = pd.concat([data, macd], axis=1)
        
        # Bollinger Bands
        bbands = ta.bbands(data['close'])
        data = pd.concat([data, bbands], axis=1)
        
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
        
        # Williams %R
        data['Williams_%R_14'] = ta.willr(data['high'], data['low'], data['close'])
        
        # Save to CSV
        csv_path = f"data/{ticker.lower()}.csv"
        data.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}, shape: {data.shape}")
    
    # Create a combined dataset (optional)
    print("Creating combined dataset...")
    combined_data = pd.DataFrame()
    
    for ticker in tickers:
        ticker_data = pd.read_csv(f"data/{ticker.lower()}.csv")
        ticker_data['ticker'] = ticker
        
        if len(combined_data) == 0:
            combined_data = ticker_data
        else:
            combined_data = pd.concat([combined_data, ticker_data])
    
    # Save combined dataset
    combined_csv_path = "data/combined_stocks.csv"
    combined_data.to_csv(combined_csv_path, index=False)
    print(f"Combined dataset saved to {combined_csv_path}, shape: {combined_data.shape}")
    
    # List available CSV files
    print("\\nAvailable data files:")
    !ls -l data/*.csv
    """))
    
    # Configuration update
    nb.cells.append(new_markdown_cell("## Update Configuration File"))
    nb.cells.append(new_code_cell("""
    import yaml
    import os
    
    # List available CSV files
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    print("Available data files:")
    for i, file in enumerate(csv_files):
        print(f"{i}: {file}")
    
    # Choose which file to use (default to the first one)
    chosen_idx = 0  # Change this to use a different file
    chosen_file = csv_files[chosen_idx]
    data_path = os.path.join('data', chosen_file)
    
    print(f"\\nUsing data file: {data_path}")
    
    # Load the default config
    config_path = 'configs/default.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update the data path
    config['data']['path'] = data_path
    
    # Save back to a new config file
    custom_config_path = 'configs/lstm_config.yaml'
    with open(custom_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated configuration saved to {custom_config_path}")
    
    # Display the updated config
    print("\\nConfiguration:")
    !cat {custom_config_path}
    """))
    
    # Project exploration
    nb.cells.append(new_markdown_cell("## Explore Project Structure"))
    nb.cells.append(new_code_cell("""
    # List important directories and files
    print("Project structure:\\n")
    !ls -la
    
    print("\\n\\nSource code files:")
    !ls -la src/
    
    print("\\n\\nModel files:")
    !ls -la src/model/
    
    print("\\n\\nData processing files:")
    !ls -la src/data/
    
    print("\\n\\nBaseline strategy files:")
    !ls -la src/baselines/
    
    print("\\n\\nConfiguration files:")
    !ls -la configs/
    """))
    
    # Dry run
    nb.cells.append(new_markdown_cell("## Run Quick Test (Dry Run)"))
    nb.cells.append(new_code_cell("""
    # Run a quick test with fewer epochs
    !python -m src.train --config configs/lstm_config.yaml --dry-run
    """))
    
    # Full training
    nb.cells.append(new_markdown_cell("## Full Model Training"))
    nb.cells.append(new_code_cell("""
    # Run full training with Kelly-based loss function
    !python -m src.train --config configs/lstm_config.yaml --save-model --model-path models/lstm_model.pt --loss-function kelly
    """))
    
    # Sequential validation
    nb.cells.append(new_markdown_cell("""
    ## Sequential Validation (Walk-Forward Testing)
    
    This is a critical step to ensure the model performs well in real-time conditions without data leakage. The sequential validation:
    
    1. Processes data point by point, exactly as in real trading
    2. Ensures no future information is used in making predictions
    3. Analyzes performance across different market regimes
    """))
    nb.cells.append(new_code_cell("""
    # Run sequential validation
    !python -m src.train --config configs/lstm_config.yaml --model-path models/lstm_model.pt --sequential-validation
    
    # Display the generated plots
    import matplotlib.pyplot as plt
    from IPython.display import display, Image
    import glob
    
    # Equity curve
    display(Image(filename='logs/sequential/equity_curve.png'))
    
    # Positions over time
    display(Image(filename='logs/sequential/positions.png'))
    
    # Position vs Future Return
    display(Image(filename='logs/sequential/position_vs_return.png'))
    
    # Return distribution
    display(Image(filename='logs/sequential/return_distribution.png'))
    """))
    
    # Market regime analysis
    nb.cells.append(new_markdown_cell("""
    ## Market Regime Analysis
    
    Analyzing how the model performs across different market conditions:
    
    * Bull markets (uptrend)
    * Bear markets (downtrend)
    * High volatility periods
    * Sideways/ranging markets
    """))
    nb.cells.append(new_code_cell("""
    # Display regime analysis plots
    regime_plots = glob.glob('logs/regimes/*.png')
    
    for plot in regime_plots:
        print(f"\\n{os.path.basename(plot)}")
        display(Image(filename=plot))
    
    # Read and display the sequential validation results
    import pandas as pd
    
    results = pd.read_csv('logs/sequential/sequential_results.csv')
    print("Performance metrics:")
    print(f"Total return: {results['total_return'].iloc[0]:.2%}")
    print(f"Sharpe ratio: {results['sharpe_ratio'].iloc[0]:.2f}")
    print(f"Max drawdown: {results['max_drawdown'].iloc[0]:.2%}")
    print(f"Win rate: {results['win_rate'].iloc[0]:.2%}")
    """))
    
    # Bootstrap validation
    nb.cells.append(new_markdown_cell("""
    ## Bootstrap Validation for Robustness Testing
    
    Bootstrap validation helps assess model robustness by:
    
    * Running multiple validation simulations on resampled data
    * Establishing confidence intervals for performance metrics
    * Identifying how consistent the model performance is
    """))
    nb.cells.append(new_code_cell("""
    # Run bootstrap validation with 20 iterations
    !python -m src.train --config configs/lstm_config.yaml --model-path models/lstm_model.pt --sequential-validation --bootstrap --bootstrap-iterations 20
    
    # Display bootstrap plots
    bootstrap_plots = glob.glob('logs/bootstrap/*.png')
    
    for plot in bootstrap_plots:
        if 'iteration' not in plot:  # Skip individual iteration plots
            print(f"\\n{os.path.basename(plot)}")
            display(Image(filename=plot))
    
    # Read and display bootstrap statistics
    bootstrap_stats = pd.read_csv('logs/bootstrap/bootstrap_stats.csv', index_col=0)
    print("\\nBootstrap Statistics:")
    display(bootstrap_stats)
    """))
    
    # Advanced comparisons
    nb.cells.append(new_markdown_cell("## Advanced Strategy Comparison"))
    nb.cells.append(new_code_cell("""
    # Compare LSTM model against traditional strategies
    !python -m src.evaluate --config configs/lstm_config.yaml --model-path models/lstm_model.pt --strategies "buy_hold,moving_average,rsi,macd" --plot
    
    # Display comparison plots
    strategy_plots = glob.glob('logs/strategy_comparison/*.png')
    
    for plot in strategy_plots:
        print(f"\\n{os.path.basename(plot)}")
        display(Image(filename=plot))
    """))
    
    # Save the notebook
    output_path = 'lstm_stock_colab_improved.ipynb'
    with open(output_path, 'w') as f:
        f.write(json.dumps(nb))
    
    print(f"Notebook saved to {output_path}")

if __name__ == "__main__":
    generate_notebook() 