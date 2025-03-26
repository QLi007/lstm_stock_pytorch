# LSTM Stock Predictor with Advanced Loss Functions

A PyTorch-based quantitative trading system with LSTM prediction and advanced loss functions, including Kelly criterion and adaptive multi-objective optimization.

## Features
- Time-series safe processing (avoiding future data leakage)
- LSTM price prediction model
- Dynamic risk control strategies 
- Advanced loss functions:
  - Kelly Enhanced Loss
  - Adaptive Multi-Objective Loss
- Comprehensive evaluation metrics
- Baseline strategy comparisons
- Automated CI/CD pipeline
- Unit tests for core functionality

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run training with default loss function
python src/train.py

# Run training with adaptive loss function
python src/train.py --loss-function adaptive --save-model 

# Run tests
python -m pytest tests/ -v
```

## Model Evaluation & Visualization

```bash
# Train model and save
python src/train.py --save-model

# Basic model evaluation
python src/evaluate.py --visualize

# Advanced evaluation with baseline comparison
python src/evaluate.py --visualize --compare-baselines --analyze-features

# Use custom parameters
python src/evaluate.py --model-path models/custom_model.pt --save-plot results/performance.png
```

## Advanced Evaluation Features

The enhanced evaluation system provides:

1. **Comprehensive Metrics**:
   - Standard prediction metrics: MSE, RMSE, MAE, Direction Accuracy
   - Risk-adjusted metrics: Sharpe, Sortino, Calmar ratios
   - Trading performance: Win Rate, Profit/Loss Ratio, Maximum Drawdown

2. **Statistical Significance Testing**:
   - t-tests for comparing means
   - Wilcoxon signed-rank tests for non-parametric comparison
   - Bootstrap confidence intervals

3. **Baseline Strategy Comparisons**:
   - Buy & Hold
   - Simple Moving Average Crossover
   - Momentum
   - Mean Reversion
   - Ensemble strategies

4. **Enhanced Visualizations**:
   - Strategy comparison charts
   - Drawdown analysis
   - Rolling performance metrics
   - Feature importance analysis

## Project Structure

```
.
├── configs/             # Configuration files
│   └── default.yaml     # Default parameter configuration
├── data/                # Data files
│   └── sample.csv       # Sample stock data
├── src/                 # Source code
│   ├── data/            # Data processing
│   │   ├── loader.py    # Data loader
│   │   └── feature_engineering.py  # Feature engineering
│   ├── model/           # Model definitions
│   │   ├── lstm.py      # LSTM model
│   │   ├── kelly_loss.py     # Kelly criterion loss function
│   │   └── advanced_loss.py  # Advanced loss functions
│   ├── baselines/       # Baseline strategies
│   │   └── simple_strategies.py  # Implementation of baseline strategies
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script with advanced metrics
├── tests/               # Unit tests
├── logs/                # Training logs (auto-created)
└── models/              # Saved models (auto-created)
```

## Loss Functions

### Kelly Enhanced Loss
Optimizes for:
- Equity curve trends
- Kelly position sizing
- Turnover penalties

### Adaptive Multi-Objective Loss
Advanced loss function that dynamically adjusts weights based on market conditions:
- Enhanced equity curve optimization
- Dynamic risk-adjusted Kelly model
- Realistic transaction cost modeling
- Adaptive weight allocation
