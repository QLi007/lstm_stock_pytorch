# 改进的基准比较代码 - 修复了之前识别的问题

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from IPython.display import display
import os
import yaml

# 基准策略类定义，统一异常处理和短序列处理
class SimpleMovingAverageCrossover:
    """简单移动平均线交叉策略"""
    def __init__(self, short_window=5, long_window=20, max_leverage=1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.max_leverage = max_leverage
        self.name = f"SMA({short_window},{long_window})"
        
    def generate_positions(self, prices):
        """生成基于SMA交叉的仓位"""
        # 统一处理短序列
        if len(prices) < self.long_window:
            return np.zeros(len(prices))
        
        short_ma = np.convolve(prices, np.ones(self.short_window)/self.short_window, mode='valid')
        long_ma = np.convolve(prices, np.ones(self.long_window)/self.long_window, mode='valid')
        
        # 对齐数组
        short_ma_aligned = short_ma[-(len(long_ma)):]
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
        self.name = f"动量({window})"
        
    def generate_positions(self, prices):
        """生成基于动量的仓位"""
        # 统一处理短序列
        if len(prices) < self.window:
            return np.zeros(len(prices))
        
        returns = np.zeros(len(prices))
        
        for i in range(self.window, len(prices)):
            momentum_return = (prices[i] / prices[i - self.window]) - 1
            
            # 应用阈值过滤
            if momentum_return > self.threshold:
                returns[i] = self.max_leverage  # 多头仓位
            elif momentum_return < -self.threshold:
                returns[i] = -self.max_leverage  # 空头仓位
        
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
        self.name = f"均值回归({window})"
        
    def generate_positions(self, prices):
        """生成基于均值回归的仓位"""
        # 统一处理短序列
        if len(prices) < self.window:
            return np.zeros(len(prices))
        
        positions = np.zeros(len(prices))
        
        for i in range(self.window, len(prices)):
            window_slice = prices[i-self.window:i]
            mean = np.mean(window_slice)
            std = np.std(window_slice)
            
            upper_band = mean + (self.num_std * std)
            lower_band = mean - (self.num_std * std)
            
            current_price = prices[i]
            
            if current_price < lower_band:
                positions[i] = self.max_leverage  # 多头仓位 - 价格低时买入
            elif current_price > upper_band:
                positions[i] = -self.max_leverage  # 空头仓位 - 价格高时卖出
            else:
                # 基于价格与均值的距离调整仓位
                distance_from_mean = (current_price - mean) / (upper_band - mean) if upper_band > mean else 0
                positions[i] = -distance_from_mean * self.max_leverage
        
        return positions


def calculate_metrics(predictions, targets, cap_extreme_values=True):
    """计算综合评估指标，增强数值稳定性"""
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
    std_return = np.std(portfolio_returns) + 1e-6  # 避免除零
    
    # 夏普比率，增加数值稳定性限制
    sharpe = mean_return / std_return * annualization_factor
    if cap_extreme_values and (np.abs(sharpe) > 10 or np.isnan(sharpe)):
        if np.isnan(sharpe):
            sharpe = 0  # 处理NaN
        else:
            sharpe = np.sign(sharpe) * min(abs(sharpe), 10)  # 限制极端值
            
    # 计算累积回报
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    
    # 最大回撤
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (peak + 1)
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # 索提诺比率（只考虑下行风险）
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = np.std(downside_returns) + 1e-6
        sortino = mean_return / downside_deviation * annualization_factor
        # 限制极端值
        if cap_extreme_values and (np.abs(sortino) > 10 or np.isnan(sortino)):
            if np.isnan(sortino):
                sortino = 0
            else:
                sortino = np.sign(sortino) * min(abs(sortino), 10)
    else:
        sortino = np.inf if mean_return > 0 else -np.inf
        if cap_extreme_values:
            sortino = 10 if mean_return > 0 else -10
    
    # 胜率
    winning_trades = np.sum(portfolio_returns > 0)
    total_trades = len(portfolio_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 结果字典
    return {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'mean_return': mean_return,
        'annualized_return': mean_return * 252,
        'std_return': std_return * np.sqrt(252),  # 年化波动率
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    }


def get_test_data_aligned(config_path):
    """加载测试数据和对应的收盘价，确保对齐"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    from src.data.loader import StockDataLoader
    loader = StockDataLoader(config_path=config_path)
    
    # 准备数据，不打乱顺序以保持时间序列性质
    # 注意：确保StockDataLoader的实现不会打乱时间序列
    train_data, val_data, test_data = loader.prepare_data(
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        shuffle=False  # 关键：不打乱数据顺序
    )
    
    # 获取测试序列
    test_sequences = loader.get_train_sequences('test')
    X_test = torch.FloatTensor(np.array([s[0] for s in test_sequences]))
    y_test = torch.FloatTensor(np.array([s[1] for s in test_sequences]))
    
    # 获取测试数据的日期和收盘价，确保它们正确对齐
    try:
        # 加载原始数据
        test_data_df = pd.read_csv(config['data']['path'])
        
        # 获取测试集的索引（基于test_size比例）
        test_start_idx = int(len(test_data_df) * (1 - config['data'].get('test_size', 0.2)))
        
        # 提取对应的收盘价和日期
        close_prices = test_data_df['close'].values[test_start_idx:]
        
        # 如果测试序列长度与收盘价长度不一致，进行调整
        if len(close_prices) != len(test_sequences):
            print(f"警告: 收盘价长度 ({len(close_prices)}) 与测试序列长度 ({len(test_sequences)}) 不一致")
            # 取共同长度的较小值
            min_len = min(len(close_prices), len(test_sequences))
            close_prices = close_prices[:min_len]
            X_test = X_test[:min_len]
            y_test = y_test[:min_len]
        
        # 提取日期（如果有）
        test_dates = test_data_df['date'].values[test_start_idx:test_start_idx + len(close_prices)] if 'date' in test_data_df.columns else None
        
    except Exception as e:
        print(f"获取收盘价数据时出错: {str(e)}")
        # 备选方案：尝试从特征数据中提取收盘价（假设这里包含收盘价）
        try:
            # 尝试找到收盘价特征的位置
            close_feature_idx = None
            if 'features' in config['data']:
                features = config['data']['features']
                # 查找可能的收盘价特征
                for i, feature in enumerate(features):
                    if 'close' in feature.lower() or 'price' in feature.lower():
                        close_feature_idx = i
                        break
            
            if close_feature_idx is not None:
                close_prices = X_test[:, -1, close_feature_idx].numpy()
            else:
                # 如果找不到收盘价特征，使用最后一个特征
                close_prices = X_test[:, -1, -1].numpy()
                print("警告: 无法确定收盘价特征位置，使用最后一个特征作为近似")
        except:
            # 最后的备选：生成随机价格（仅用于测试）
            print("警告: 无法提取收盘价，生成模拟价格数据")
            close_prices = np.linspace(100, 200, len(X_test)) + np.random.normal(0, 5, len(X_test))
        
        test_dates = None
    
    return X_test, y_test, close_prices, test_dates


def load_trained_model(model_path, input_size, hidden_size):
    """加载训练好的模型，包含健壮的错误处理"""
    from src.model.lstm import LSTMPredictor
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到训练好的模型 {model_path}")
        return None
    
    try:
        # 初始化模型
        model = LSTMPredictor(
            input_size=input_size, 
            hidden_size=hidden_size
        )
        
        # 加载模型权重
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"模型已加载: {model_path}")
        return model
    
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None


def run_baseline_comparison(model=None, X_test=None, y_test=None, close_prices=None, test_dates=None, 
                            strategy_params=None):
    """
    运行基准比较，支持自定义策略参数
    
    Args:
        model: 训练好的LSTM模型
        X_test: 测试特征数据
        y_test: 测试标签数据
        close_prices: 测试数据的收盘价序列
        test_dates: 测试数据的日期
        strategy_params: 策略参数字典，例如 {'sma': {'short_window': 5, 'long_window': 20}}
    """
    # 验证输入
    if model is None or X_test is None or y_test is None:
        print("错误: 缺少模型或测试数据")
        return None
    
    if close_prices is None:
        print("警告: 缺少收盘价数据，无法运行基准策略比较")
        return None
    
    # 设置默认策略参数
    default_params = {
        'sma': {'short_window': 5, 'long_window': 20, 'max_leverage': 1.0},
        'momentum': {'window': 10, 'max_leverage': 1.0, 'threshold': 0.0},
        'mean_reversion': {'window': 20, 'num_std': 2.0, 'max_leverage': 1.0}
    }
    
    # 使用用户提供的参数覆盖默认参数
    if strategy_params is not None:
        for strategy, params in strategy_params.items():
            if strategy in default_params:
                default_params[strategy].update(params)
    
    # 获取模型预测
    try:
        with torch.no_grad():
            predictions = model(X_test).squeeze().detach().cpu().numpy()
        
        true_returns = y_test.cpu().numpy()
    except Exception as e:
        print(f"模型预测出错: {str(e)}")
        return None
    
    # 计算LSTM模型回报
    model_returns = predictions * true_returns
    
    # 基准策略1：买入并持有
    buyhold_returns = true_returns
    
    # 策略返回和性能指标
    all_returns = {'LSTM模型': model_returns, '买入并持有': buyhold_returns}
    all_metrics = {}
    
    # 创建和运行所有基准策略
    strategies = {
        'sma': SimpleMovingAverageCrossover(**default_params['sma']),
        'momentum': MomentumStrategy(**default_params['momentum']),
        'mean_reversion': MeanReversionStrategy(**default_params['mean_reversion'])
    }
    
    for strategy_key, strategy in strategies.items():
        try:
            positions = strategy.generate_positions(close_prices)
            strategy_returns = positions * true_returns
            all_returns[strategy.name] = strategy_returns
        except Exception as e:
            print(f"{strategy.name}策略出错: {str(e)}")
            all_returns[strategy.name] = np.zeros_like(true_returns)
    
    # 计算所有策略的累积回报
    cumulative_returns = {}
    for name, returns in all_returns.items():
        cumulative_returns[name] = np.cumprod(1 + returns) - 1
        all_metrics[name] = calculate_metrics(
            returns / np.abs(returns).max() if np.abs(returns).max() > 0 else returns,  # 归一化返回用于指标计算
            true_returns
        )
    
    # 绘制累积回报比较图
    plt.figure(figsize=(14, 7))
    
    for name, returns in cumulative_returns.items():
        plt.plot(returns, label=f'{name} (Sharpe: {all_metrics[name]["sharpe"]:.2f})')
    
    plt.title('策略比较: 累积回报')
    plt.xlabel('时间')
    plt.ylabel('累积回报')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if test_dates is not None:
        # 显示部分日期标签
        n_ticks = min(10, len(test_dates))
        tick_indices = np.linspace(0, len(test_dates)-1, n_ticks, dtype=int)
        plt.xticks(tick_indices, [test_dates[i] for i in tick_indices], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 创建性能指标比较表格
    metrics_to_show = ['sharpe', 'sortino', 'max_drawdown', 'win_rate', 'direction_accuracy', 'final_return']
    metrics_names = ['夏普比率', '索提诺比率', '最大回撤', '胜率', '方向准确率', '最终回报']
    
    metrics_data = {}
    for name, metrics in all_metrics.items():
        metrics_data[name] = [metrics.get(m, "N/A") for m in metrics_to_show]
    
    metrics_df = pd.DataFrame(metrics_data, index=metrics_names)
    
    # 显示表格，使用样式高亮最佳/最差值
    # 对于夏普比率、索提诺比率、胜率、方向准确率和最终回报，值越高越好
    # 对于最大回撤，值越低越好
    display(metrics_df.style
            .format("{:.4f}")
            .highlight_max(subset=metrics_df.index.difference(['最大回撤']), axis=1, color='lightgreen')
            .highlight_min(subset=['最大回撤'], axis=1, color='lightgreen')
            .highlight_min(subset=metrics_df.index.difference(['最大回撤']), axis=1, color='lightcoral')
            .highlight_max(subset=['最大回撤'], axis=1, color='lightcoral'))
    
    # 统计显著性测试
    try:
        t_stat, p_value = stats.ttest_ind(model_returns, buyhold_returns, equal_var=False)
        print(f"\n统计显著性测试 (LSTM vs 买入并持有):")
        print(f"t值: {t_stat:.4f}, p值: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
        significant = p_value < 0.05 and t_stat > 0
        print(f"在95%置信水平下LSTM模型{'优于' if significant else '不优于'}买入并持有策略")
    except Exception as e:
        print(f"统计显著性测试出错: {str(e)}")
    
    # 绘制策略回报直方图比较
    plt.figure(figsize=(14, 7))
    bins = np.linspace(min(model_returns.min(), buyhold_returns.min()) * 1.1,
                       max(model_returns.max(), buyhold_returns.max()) * 1.1, 50)
    
    plt.hist(model_returns, bins=bins, alpha=0.5, label='LSTM模型', density=True)
    plt.hist(buyhold_returns, bins=bins, alpha=0.5, label='买入并持有', density=True)
    
    plt.title('回报分布比较')
    plt.xlabel('每日回报')
    plt.ylabel('频率密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'metrics': all_metrics,
        'cumulative_returns': cumulative_returns,
        'returns': all_returns
    }


# 使用示例（添加到Colab笔记本中）
"""
# 设置策略参数（可选）
strategy_params = {
    'sma': {'short_window': 5, 'long_window': 20},
    'momentum': {'window': 10, 'threshold': 0.01},
    'mean_reversion': {'window': 20, 'num_std': 2.0}
}

# 获取对齐的测试数据
X_test, y_test, close_prices, test_dates = get_test_data_aligned(config_path)

# 加载模型
trained_model = load_trained_model(
    model_path="models/lstm_model.pt",
    input_size=X_test.shape[2],
    hidden_size=128
)

# 如果模型加载成功，运行基准比较
if trained_model is not None:
    comparison_results = run_baseline_comparison(
        model=trained_model,
        X_test=X_test,
        y_test=y_test,
        close_prices=close_prices,
        test_dates=test_dates,
        strategy_params=strategy_params
    )
    print("基准比较完成!")
else:
    print("找不到训练好的模型，请先完成训练!")
""" 