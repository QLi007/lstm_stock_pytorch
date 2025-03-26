# 在训练完成后添加此单元格来进行基准策略比较
# 确保已经训练并保存了模型

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from IPython.display import display

class SimpleMovingAverageCrossover:
    def __init__(self, short_window=5, long_window=20, max_leverage=1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.max_leverage = max_leverage
        
    def generate_positions(self, prices):
        if len(prices) < self.long_window:
            return np.zeros(len(prices))
        
        short_ma = np.convolve(prices, np.ones(self.short_window)/self.short_window, mode='valid')
        long_ma = np.convolve(prices, np.ones(self.long_window)/self.long_window, mode='valid')
        short_ma_aligned = short_ma[-(len(long_ma)):]
        positions = np.zeros(len(prices))
        
        for i in range(len(long_ma)):
            idx = i + (len(prices) - len(long_ma))
            if short_ma_aligned[i] > long_ma[i]:
                positions[idx] = self.max_leverage
            elif short_ma_aligned[i] < long_ma[i]:
                positions[idx] = -self.max_leverage
        
        if len(long_ma) > 0:
            positions[:len(prices) - len(long_ma)] = positions[len(prices) - len(long_ma)]
        return positions

class MomentumStrategy:
    def __init__(self, window=10, max_leverage=1.0, threshold=0.0):
        self.window = window
        self.max_leverage = max_leverage
        self.threshold = threshold
        
    def generate_positions(self, prices):
        if len(prices) < self.window:
            return np.zeros(len(prices))
        
        returns = np.zeros(len(prices))
        for i in range(self.window, len(prices)):
            momentum_return = (prices[i] / prices[i - self.window]) - 1
            if momentum_return > self.threshold:
                returns[i] = self.max_leverage
            elif momentum_return < -self.threshold:
                returns[i] = -self.max_leverage
        
        first_valid = next((i for i, r in enumerate(returns) if r != 0), None)
        if first_valid is not None:
            returns[:first_valid] = returns[first_valid]
        return returns

# 当模型训练完毕后，使用以下代码进行基准比较

# 加载模型和测试数据
# 假设已经有以下变量:
# - trained_model: 训练好的LSTM模型
# - X_test: 测试特征数据
# - y_test: 测试标签数据
# - test_prices: 测试数据的收盘价序列（用于基准策略）

# 如果没有，请先进行如下准备：
# trained_model = LSTMPredictor(...)
# trained_model.load_state_dict(torch.load(model_path))
# X_test = ...
# y_test = ...
# test_prices = ...  # 收盘价序列

def run_baseline_comparison():
    # 确保已经有这些变量
    if 'trained_model' not in globals() or 'X_test' not in globals() or 'y_test' not in globals():
        print("请先准备模型和测试数据。训练模型并将其保存为 trained_model，测试数据为 X_test 和 y_test。")
        return
    
    # 获取收盘价序列（如果没有提供）
    if 'test_prices' not in globals():
        if isinstance(X_test, torch.Tensor):
            # 假设最后一个特征是收盘价
            test_prices = X_test[:, -1, -1].cpu().numpy()
        else:
            test_prices = X_test[:, -1, -1]
    
    # 模型预测
    trained_model.eval()
    with torch.no_grad():
        if isinstance(X_test, torch.Tensor):
            predictions = trained_model(X_test).squeeze().detach().cpu().numpy()
        else:
            predictions = trained_model(torch.FloatTensor(X_test)).squeeze().detach().cpu().numpy()
    
    if isinstance(y_test, torch.Tensor):
        true_returns = y_test.cpu().numpy()
    else:
        true_returns = y_test
    
    # 计算LSTM模型回报
    model_returns = predictions * true_returns
    
    # 基准策略1：买入持有
    buyhold_returns = true_returns
    
    # 基准策略2：SMA交叉
    sma_strategy = SimpleMovingAverageCrossover(short_window=5, long_window=20)
    sma_positions = sma_strategy.generate_positions(test_prices)
    sma_returns = sma_positions * true_returns
    
    # 基准策略3：动量策略
    momentum_strategy = MomentumStrategy(window=10)
    momentum_positions = momentum_strategy.generate_positions(test_prices)
    momentum_returns = momentum_positions * true_returns
    
    # 计算累积回报
    cumulative_model = np.cumprod(1 + model_returns) - 1
    cumulative_buyhold = np.cumprod(1 + buyhold_returns) - 1
    cumulative_sma = np.cumprod(1 + sma_returns) - 1
    cumulative_momentum = np.cumprod(1 + momentum_returns) - 1
    
    # 计算评估指标
    def calc_metrics(returns):
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        cumulative = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        win_rate = np.sum(returns > 0) / len(returns)
        return {
            'sharpe': sharpe, 
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_return': cumulative[-1] if len(cumulative) > 0 else 0
        }
    
    model_metrics = calc_metrics(model_returns)
    buyhold_metrics = calc_metrics(buyhold_returns)
    sma_metrics = calc_metrics(sma_returns)
    momentum_metrics = calc_metrics(momentum_returns)
    
    # 绘制累积回报比较图
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_model, label=f'LSTM模型 (Sharpe: {model_metrics["sharpe"]:.2f})')
    plt.plot(cumulative_buyhold, label=f'买入持有 (Sharpe: {buyhold_metrics["sharpe"]:.2f})')
    plt.plot(cumulative_sma, label=f'SMA交叉 (Sharpe: {sma_metrics["sharpe"]:.2f})')
    plt.plot(cumulative_momentum, label=f'动量策略 (Sharpe: {momentum_metrics["sharpe"]:.2f})')
    plt.title('策略比较: 累积回报')
    plt.xlabel('时间')
    plt.ylabel('累积回报')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 创建性能指标比较表格
    metrics_df = pd.DataFrame({
        'LSTM模型': [model_metrics['sharpe'], model_metrics['max_drawdown'], model_metrics['win_rate'], model_metrics['final_return']],
        '买入持有': [buyhold_metrics['sharpe'], buyhold_metrics['max_drawdown'], buyhold_metrics['win_rate'], buyhold_metrics['final_return']],
        'SMA交叉': [sma_metrics['sharpe'], sma_metrics['max_drawdown'], sma_metrics['win_rate'], sma_metrics['final_return']],
        '动量策略': [momentum_metrics['sharpe'], momentum_metrics['max_drawdown'], momentum_metrics['win_rate'], momentum_metrics['final_return']]
    }, index=['夏普比率', '最大回撤', '胜率', '最终回报'])
    
    display(metrics_df.style.format("{:.4f}").highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='lightcoral'))
    
    # 统计显著性测试 (t-test)
    t_stat, p_value = stats.ttest_ind(model_returns, buyhold_returns, equal_var=False)
    print(f"\n统计显著性测试 (LSTM vs 买入持有):")
    print(f"t值: {t_stat:.4f}, p值: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
    print(f"在95%置信水平下LSTM模型{'优于' if p_value < 0.05 and t_stat > 0 else '不优于'}买入持有策略")

# 运行比较
# run_baseline_comparison()  # 取消注释此行运行比较 