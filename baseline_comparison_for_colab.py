# 添加此代码块到Colab笔记本中以实现基准策略比较

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats

# 定义基准策略类
class SimpleMovingAverageCrossover:
    """简单移动平均线交叉策略"""
    def __init__(self, short_window=5, long_window=20, max_leverage=1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.max_leverage = max_leverage
        
    def generate_positions(self, prices):
        if len(prices) < self.long_window:
            raise ValueError(f"价格数据太短，至少需要{self.long_window}个点。")
        
        # 计算短期和长期移动平均线
        short_ma = np.convolve(prices, np.ones(self.short_window)/self.short_window, mode='valid')
        long_ma = np.convolve(prices, np.ones(self.long_window)/self.long_window, mode='valid')
        
        # 对齐数组
        short_ma_aligned = short_ma[-(len(long_ma)):]
        
        # 生成信号
        positions = np.zeros(len(prices))
        
        # 根据交叉策略填充仓位
        for i in range(len(long_ma)):
            idx = i + (len(prices) - len(long_ma))
            if short_ma_aligned[i] > long_ma[i]:
                positions[idx] = self.max_leverage  # 多头仓位
            elif short_ma_aligned[i] < long_ma[i]:
                positions[idx] = -self.max_leverage  # 空头仓位
        
        # 基于早期信号填充初始仓位
        if len(long_ma) > 0:
            positions[:len(prices) - len(long_ma)] = positions[len(prices) - len(long_ma)]
        
        return positions

class MomentumStrategy:
    """动量交易策略"""
    def __init__(self, window=10, max_leverage=1.0, threshold=0.0):
        self.window = window
        self.max_leverage = max_leverage
        self.threshold = threshold
        
    def generate_positions(self, prices):
        if len(prices) < self.window:
            raise ValueError(f"价格数据太短，至少需要{self.window}个点。")
        
        # 计算回报
        returns = np.zeros(len(prices))
        
        for i in range(self.window, len(prices)):
            momentum_return = (prices[i] / prices[i - self.window]) - 1
            
            # 应用阈值过滤
            if momentum_return > self.threshold:
                returns[i] = self.max_leverage  # 多头仓位
            elif momentum_return < -self.threshold:
                returns[i] = -self.max_leverage  # 空头仓位
            else:
                returns[i] = 0  # 无仓位
        
        # 基于第一个有效信号填充初始仓位
        first_valid = next((i for i, r in enumerate(returns) if r != 0), None)
        if first_valid is not None:
            returns[:first_valid] = returns[first_valid]
        
        return returns

class MeanReversionStrategy:
    """均值回归策略"""
    def __init__(self, window=20, num_std=2.0, max_leverage=1.0):
        self.window = window
        self.num_std = num_std
        self.max_leverage = max_leverage
        
    def generate_positions(self, prices):
        if len(prices) < self.window:
            raise ValueError(f"价格数据太短，至少需要{self.window}个点。")
        
        positions = np.zeros(len(prices))
        
        for i in range(self.window, len(prices)):
            window_slice = prices[i-self.window:i]
            mean = np.mean(window_slice)
            std = np.std(window_slice)
            
            upper_band = mean + (self.num_std * std)
            lower_band = mean - (self.num_std * std)
            
            current_price = prices[i]
            
            if current_price < lower_band:
                positions[i] = self.max_leverage  # 多头仓位
            elif current_price > upper_band:
                positions[i] = -self.max_leverage  # 空头仓位
            else:
                # 基于价格与均值的距离调整仓位
                distance_from_mean = (current_price - mean) / (upper_band - mean)
                positions[i] = -distance_from_mean * self.max_leverage
        
        # 填充初始仓位
        positions[:self.window] = 0
        
        return positions

def calculate_metrics(predictions, targets):
    """计算综合评估指标"""
    # 转换为numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # 基本预测准确性指标
    mse = np.mean((targets - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(targets - predictions))
    
    # 计算方向准确率
    correct_direction = np.sum(np.sign(predictions) == np.sign(targets))
    direction_accuracy = correct_direction / len(targets) if len(targets) > 0 else 0
    
    # 投资组合回报
    portfolio_returns = predictions * targets
    
    # 风险调整回报指标
    annualization_factor = np.sqrt(252)  # 对于日数据
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns) + 1e-6
    
    # 夏普比率
    sharpe = mean_return / std_return * annualization_factor
    
    # 计算累积回报
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    
    # 最大回撤
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (peak + 1)
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # 卡尔玛比率
    calmar = (mean_return * 252) / max_drawdown if max_drawdown > 0 else float('inf')
    
    # 索提诺比率
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = np.std(downside_returns) + 1e-6
        sortino = mean_return / downside_deviation * annualization_factor
    else:
        sortino = float('inf')
    
    # 胜率
    winning_trades = np.sum(portfolio_returns > 0)
    total_trades = len(portfolio_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 盈亏比
    avg_profit = np.mean(portfolio_returns[portfolio_returns > 0]) if np.any(portfolio_returns > 0) else 0
    avg_loss = np.abs(np.mean(portfolio_returns[portfolio_returns < 0])) if np.any(portfolio_returns < 0) else 1e-6
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
    
    # 返回结果字典
    return {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'mean_return': mean_return,
        'annualized_return': mean_return * 252,
        'std_return': std_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'final_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    }

def statistical_significance_test(strategy_returns, benchmark_returns):
    """执行统计显著性测试"""
    # t检验
    t_stat, p_value_t = stats.ttest_ind(strategy_returns, benchmark_returns, equal_var=False)
    
    # Wilcoxon符号秩检验
    try:
        w_stat, p_value_w = stats.wilcoxon(strategy_returns, benchmark_returns)
    except:
        w_stat, p_value_w = 0, 1.0
    
    # Bootstrap分析
    n_bootstrap = 1000
    bootstrap_differences = []
    
    for _ in range(n_bootstrap):
        strategy_sample = np.random.choice(strategy_returns, size=len(strategy_returns), replace=True)
        benchmark_sample = np.random.choice(benchmark_returns, size=len(benchmark_returns), replace=True)
        bootstrap_differences.append(np.mean(strategy_sample) - np.mean(benchmark_sample))
    
    # Bootstrap置信区间
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

def compare_with_baselines(model, X, y, close_prices=None, data_dates=None):
    """
    将LSTM模型与基准策略进行比较
    
    Args:
        model: 训练好的LSTM模型
        X: 特征矩阵 (PyTorch张量或numpy数组)
        y: 目标回报 (PyTorch张量或numpy数组) 
        close_prices: 收盘价数组，用于基准策略 (如果为None，使用X的最后一个特征)
        data_dates: 对应数据点的日期 (如果可用)
    
    Returns:
        比较结果字典
    """
    # 获取模型预测
    if isinstance(X, torch.Tensor):
        model.eval()
        with torch.no_grad():
            model_predictions = model(X).squeeze().detach().cpu().numpy()
    else:
        model.eval()
        with torch.no_grad():
            model_predictions = model(torch.FloatTensor(X)).squeeze().detach().cpu().numpy()
    
    if isinstance(y, torch.Tensor):
        actual_returns = y.cpu().numpy()
    else:
        actual_returns = y
    
    # 计算模型性能
    model_performance = calculate_metrics(model_predictions, actual_returns)
    model_returns = model_predictions * actual_returns
    
    # 买入并持有策略
    buyhold_performance = calculate_metrics(np.ones_like(actual_returns), actual_returns)
    buyhold_returns = actual_returns
    
    # 获取收盘价数据
    if close_prices is None:
        if isinstance(X, torch.Tensor):
            # 假设最后一个特征是收盘价
            close_prices = X[:, -1, -1].cpu().numpy()
        else:
            close_prices = X[:, -1, -1]
    
    # SMA交叉策略
    sma_strategy = SimpleMovingAverageCrossover(short_window=5, long_window=20)
    try:
        sma_positions = sma_strategy.generate_positions(close_prices)
        sma_performance = calculate_metrics(sma_positions, actual_returns)
        sma_returns = sma_positions * actual_returns
    except Exception as e:
        print(f"SMA策略错误: {str(e)}")
        sma_performance = {"error": str(e)}
        sma_returns = np.zeros_like(actual_returns)
    
    # 动量策略
    momentum_strategy = MomentumStrategy(window=10)
    try:
        momentum_positions = momentum_strategy.generate_positions(close_prices)
        momentum_performance = calculate_metrics(momentum_positions, actual_returns)
        momentum_returns = momentum_positions * actual_returns
    except Exception as e:
        print(f"动量策略错误: {str(e)}")
        momentum_performance = {"error": str(e)}
        momentum_returns = np.zeros_like(actual_returns)
    
    # 均值回归策略
    mean_reversion_strategy = MeanReversionStrategy(window=20)
    try:
        mean_reversion_positions = mean_reversion_strategy.generate_positions(close_prices)
        mean_reversion_performance = calculate_metrics(mean_reversion_positions, actual_returns)
        mean_reversion_returns = mean_reversion_positions * actual_returns
    except Exception as e:
        print(f"均值回归策略错误: {str(e)}")
        mean_reversion_performance = {"error": str(e)}
        mean_reversion_returns = np.zeros_like(actual_returns)
    
    # 统计显著性测试
    model_vs_buyhold = statistical_significance_test(model_returns, buyhold_returns)
    model_vs_sma = statistical_significance_test(model_returns, sma_returns)
    model_vs_momentum = statistical_significance_test(model_returns, momentum_returns)
    
    # 创建可视化
    strategies = {
        "LSTM模型": model_returns,
        "买入并持有": buyhold_returns,
        "SMA交叉策略": sma_returns,
        "动量策略": momentum_returns,
        "均值回归策略": mean_reversion_returns
    }
    
    # 计算每个策略的累积回报
    cum_returns = {}
    for name, returns in strategies.items():
        cum_returns[name] = np.cumprod(1 + returns) - 1
    
    # 绘制比较图
    plt.figure(figsize=(14, 8))
    
    for name, returns in cum_returns.items():
        plt.plot(returns, label=name)
    
    plt.title('策略比较: 累积回报')
    plt.xlabel('时间')
    plt.ylabel('累积回报')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加日期标签
    if data_dates is not None and len(data_dates) == len(actual_returns):
        n_ticks = min(10, len(data_dates))
        tick_indices = np.linspace(0, len(data_dates)-1, n_ticks, dtype=int)
        plt.xticks(tick_indices, [data_dates[i] for i in tick_indices], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 创建指标比较表格
    metrics_to_show = ['sharpe', 'sortino', 'calmar', 'max_drawdown', 'win_rate', 'final_return']
    strategies_perf = {
        "LSTM模型": model_performance,
        "买入并持有": buyhold_performance,
        "SMA交叉策略": sma_performance,
        "动量策略": momentum_performance,
        "均值回归策略": mean_reversion_performance
    }
    
    # 创建比较表格
    metrics_df = pd.DataFrame(index=metrics_to_show)
    
    for strategy_name, performance in strategies_perf.items():
        if isinstance(performance, dict) and "error" not in performance:
            metrics_df[strategy_name] = [performance.get(metric, "N/A") for metric in metrics_to_show]
        else:
            metrics_df[strategy_name] = ["N/A"] * len(metrics_to_show)
    
    # 格式化指标名称
    metrics_df.index = [
        "夏普比率", "索提诺比率", "卡尔玛比率", "最大回撤", "胜率", "最终回报"
    ]
    
    # 显示表格
    display(metrics_df.style.format("{:.4f}").highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='lightcoral'))
    
    # 打印统计显著性结果
    print("\n统计显著性测试结果 (LSTM vs 买入并持有):")
    print(f"t检验 p值: {model_vs_buyhold['p_value_t']:.4f} {'*' if model_vs_buyhold['p_value_t'] < 0.05 else ''}")
    print(f"Bootstrap 95%置信区间: [{model_vs_buyhold['bootstrap_95ci_lower']:.4f}, {model_vs_buyhold['bootstrap_95ci_upper']:.4f}]")
    print(f"95%置信区间内统计显著优于基准: {'是' if model_vs_buyhold['outperforms_95ci'] else '否'}")
    
    return {
        "model_performance": model_performance,
        "buyhold_performance": buyhold_performance,
        "sma_performance": sma_performance,
        "momentum_performance": momentum_performance,
        "mean_reversion_performance": mean_reversion_performance,
        "statistical_tests": {
            "model_vs_buyhold": model_vs_buyhold,
            "model_vs_sma": model_vs_sma,
            "model_vs_momentum": model_vs_momentum
        }
    } 