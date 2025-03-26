# LSTM Stock Prediction

A PyTorch implementation of LSTM-based stock prediction with sequential validation and residual blocks.

## Features
- LSTM model with residual blocks for better training
- Custom Kelly Criterion + Drawdown loss function
- Sequential validation with walk-forward testing
- Market regime analysis
- GPU support (optional)
- Technical indicators and data preprocessing

## Setup

### Prerequisites
- Python 3.8 or higher
- PowerShell (for Windows)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/QLi007/lstm_stock_pytorch.git
cd lstm_stock_pytorch
```

2. Set up the virtual environment:
```powershell
.\setup_env.ps1
```

3. Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

## Usage

### Running the Script

1. Make sure the virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

2. Run the main script:
```powershell
python lstm_stock_colab_standalone.py
```

### Using in Google Colab

1. Open `lstm_stock_colab_standalone.ipynb` in Google Colab
2. Go to Runtime > Change runtime type
3. Select "GPU" as Hardware accelerator (optional)
4. Click "Save"
5. Run all cells

## Project Structure
```
lstm_stock_pytorch/
├── data/               # Stock data storage
├── models/            # Saved model checkpoints
├── results/           # Training results and plots
├── logs/              # Training logs
├── venv/              # Virtual environment
├── requirements.txt   # Python dependencies
├── setup_env.ps1      # Environment setup script
├── README.md          # This file
└── lstm_stock_colab_standalone.py  # Main script
```

## Model Architecture
- LSTM with residual blocks
- Attention mechanism
- Custom Kelly Criterion + Drawdown loss
- GPU optimization (when available)

## License
MIT License 