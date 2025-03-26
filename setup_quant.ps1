# 保存为 create_project.ps1
# 右键 "使用 PowerShell 运行"

# 配置项目参数
$projectName = "lstm_stock_pytorch"
$baseDir = Join-Path -Path $PSScriptRoot -ChildPath $projectName

# 清理旧目录（如果存在）
if (Test-Path $baseDir) {
    Remove-Item $baseDir -Recurse -Force
}

# 创建完整目录结构
$dirs = @(
    "$baseDir/.github/workflows",
    "$baseDir/configs",
    "$baseDir/data",
    "$baseDir/src/data",
    "$baseDir/src/model",
    "$baseDir/tests",
    "$baseDir/docs",
    "$baseDir/logs"
)

New-Item -Path $dirs -ItemType Directory -Force | Out-Null

# ==================== 生成所有核心文件 ====================

# 1. 数据加载器 (src/data/loader.py)
@"
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class StockDataLoader:
    def __init__(self, config_path='configs/default.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.data_path = Path(self.config['data']['path'])
        self.seq_length = self.config['training']['seq_length']
        
    def _safe_load_data(self):
        """严格避免未来函数的数据加载方法"""
        df = pd.read_csv(self.data_path, parse_dates=['date'])
        df = df.sort_values('date')
        
        # 滞后特征计算
        df['prev_close'] = df['close'].shift(1)
        df['MA5'] = df['prev_close'].rolling(5).mean()
        df['RSI'] = self._calculate_rsi(df['prev_close'])
        df['MACD'] = self._calculate_macd(df['prev_close'])
        
        # 目标变量
        df['future_5d_return'] = df['close'].pct_change(5).shift(-5)
        
        return df.dropna()
    
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-6)))
    
    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    def get_train_data(self):
        df = self._safe_load_data()
        features = df[self.config['data']['features']].values
        targets = df[self.config['data']['target']].values
        
        sequences = []
        for i in range(len(features) - self.seq_length - 5):
            seq_features = features[i:i+self.seq_length]
            seq_target = targets[i+self.seq_length+4]  # 预测5天后的收益
            sequences.append((seq_features, seq_target))
            
        return np.array(sequences, dtype=object)
"@ | Out-File -Encoding utf8 "$baseDir/src/data/loader.py"

# 2. LSTM模型 (src/model/lstm.py)
@"
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Tanh()  # 输出范围[-1, 1]表示仓位比例
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后一个时间步
"@ | Out-File -Encoding utf8 "$baseDir/src/model/lstm.py"

# 3. 凯利损失函数 (src/model/kelly_loss.py)
@"
import torch

class KellyLoss(torch.nn.Module):
    def __init__(self, max_leverage=2.0, alpha=0.7):
        super().__init__()
        self.max_leverage = max_leverage
        self.alpha = alpha
        
    def forward(self, preds, targets):
        portfolio_returns = preds * targets
        mu = torch.mean(portfolio_returns)
        sigma = torch.std(portfolio_returns)
        
        # 动态凯利系数计算
        kelly_f = torch.clamp(mu / (sigma**2 + 1e-6), 0.0, self.max_leverage)
        
        # 组合损失项
        position_loss = torch.mean((preds - kelly_f)**2)
        return_penalty = -mu  # 负号表示最大化收益
        
        return self.alpha * position_loss + (1 - self.alpha) * return_penalty
"@ | Out-File -Encoding utf8 "$baseDir/src/model/kelly_loss.py"

# 4. 训练脚本 (src/train.py)
@"
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.data.loader import StockDataLoader
from src.model.lstm import LSTMPredictor
from src.model.kelly_loss import KellyLoss

def main():
    # 加载配置
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    
    # 数据准备
    loader = StockDataLoader()
    sequences = loader.get_train_data()
    
    # 转换为Tensor
    X = torch.FloatTensor(np.array([s[0] for s in sequences]))
    y = torch.FloatTensor(np.array([s[1] for s in sequences]))
    dataset = TensorDataset(X, y)
    
    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0  # Windows需设为0
    )
    
    # 初始化模型
    model = LSTMPredictor(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size']
    )
    
    # 损失函数和优化器
    criterion = KellyLoss(
        max_leverage=config['training']['max_leverage'],
        alpha=0.7
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # 训练循环
    print("开始训练...")
    for epoch in range(config['training']['epochs']):
        total_loss = 0.0
        model.train()
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
"@ | Out-File -Encoding utf8 "$baseDir/src/train.py"

# 5. 配置文件 (configs/default.yaml)
@"
training:
  batch_size: 64
  seq_length: 30
  epochs: 100
  learning_rate: 0.001
  max_leverage: 2.0

data:
  path: "data/sample.csv"
  features: ["prev_close", "MA5", "RSI", "MACD"]
  target: "future_5d_return"

model:
  input_size: 4
  hidden_size: 64
"@ | Out-File -Encoding utf8 "$baseDir/configs/default.yaml"

# 6. 单元测试 (tests/test_model.py)
@"
import torch
import pytest
from src.model.lstm import LSTMPredictor
from src.model.kelly_loss import KellyLoss

def test_lstm_forward_pass():
    model = LSTMPredictor(input_size=4)
    x = torch.randn(32, 30, 4)  # (batch, seq_len, features)
    out = model(x)
    assert out.shape == (32, 1), "输出形状应为(batch_size, 1)"
    assert torch.all(out >= -1) and torch.all(out <= 1), "输出应在[-1, 1]范围内"

def test_kelly_loss_calculation():
    criterion = KellyLoss(max_leverage=2.0)
    preds = torch.tensor([0.8, 1.2, -0.5], dtype=torch.float32)
    targets = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float32)
    loss = criterion(preds, targets)
    assert loss.item() > 0, "损失值应为正数"
    assert not torch.isnan(loss), "损失值不应为NaN"
"@ | Out-File -Encoding utf8 "$baseDir/tests/test_model.py"

# 7. CI/CD配置 (.github/workflows/ci.yml)
@"
name: Quant Trading CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: Run unit tests
        run: |
          python -m pytest tests/ -v
          
      - name: Training dry-run
        run: |
          python src/train.py --dry-run
"@ | Out-File -Encoding utf8 "$baseDir/.github/workflows/ci.yml"

# 8. 依赖文件 (requirements.txt)
@"
torch>=2.0.1
pandas>=2.0.3
numpy>=1.24.4
PyYAML>=6.0
scikit-learn>=1.3.0
pytest>=7.4.0
"@ | Out-File -Encoding utf8 "$baseDir/requirements.txt"

# 9. README文档 (README.md)
@"
# LSTM Stock Predictor with Kelly Criterion

基于PyTorch的量化交易系统，集成动态凯利公式仓位管理

## 功能特性
- 时间序列安全处理（严格避免未来函数）
- LSTM价格预测模型
- 动态风险控制策略
- 自动化CI/CD流水线
- 单元测试覆盖核心功能

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行训练
python src/train.py

# 执行测试
python -m pytest tests/ -v
```
"@ | Out-File -Encoding utf8 "$baseDir/README.md"
