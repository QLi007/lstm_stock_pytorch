# 添加此单元格到Colab笔记本中，修复了之前识别的所有问题

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from IPython.display import display
import os
import yaml
from src.data.loader import StockDataLoader
from src.model.lstm import LSTMPredictor

# 定义基准策略类
class SimpleMovingAverageCrossover:
    """简单移动平均线交叉策略"""
    def __init__(self, short_window=5, long_window=20, max_leverage=1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.max_leverage = max_leverage
        self.name = f"SMA({short_window},{long_window})"
        
    def generate_positions(self, prices):
        """生成基于SMA交叉的仓位"""
        # 处理短序列
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
    """动量交易策略"""
    def __init__(self, window=10, max_leverage=1.0, threshold=0.0):
        self.window = window
        self.max_leverage = max_leverage
        self.threshold = threshold
        self.name = f"动量({window})"
        
    def generate_positions(self, prices):
        """生成基于动量的仓位"""
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

class MeanReversionStrategy:
    """均值回归策略"""
    def __init__(self, window=20, num_std=2.0, max_leverage=1.0):
        self.window = window
        self.num_std = num_std
        self.max_leverage = max_leverage
        self.name = f"均值回归({window})"
        
    def generate_positions(self, prices):
        """生成基于均值回归的仓位"""
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
                positions[i] = self.max_leverage
            elif current_price > upper_band:
                positions[i] = -self.max_leverage
            else:
                distance_from_mean = (current_price - mean) / (upper_band - mean) if upper_band > mean else 0
                positions[i] = -distance_from_mean * self.max_leverage
        
        return positions

# 运行基准比较的函数
def run_baseline_comparison():
    """运行基准比较分析"""
    
    # 检查是否有必要的变量
    if 'config_path' not in globals():
        print("错误: 找不到配置文件路径。请先设置 config_path 变量。")
        return
    
    # 获取测试数据
    print("加载数据中...")
    try:
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 准备数据，不打乱顺序以保持时间序列性质
        loader = StockDataLoader(config_path=config_path)
        train_data, val_data, test_data = loader.prepare_data(
            test_size=config['data'].get('test_size', 0.2),
            val_size=config['data'].get('val_size', 0.1),
            shuffle=False  # 不打乱数据顺序
        )
        
        # 获取测试序列
        test_sequences = loader.get_train_sequences('test')
        X_test = torch.FloatTensor(np.array([s[0] for s in test_sequences]))
        y_test = torch.FloatTensor(np.array([s[1] for s in test_sequences]))
        
        # 获取对齐的收盘价
        test_data_df = pd.read_csv(config['data']['path'])
        test_start_idx = int(len(test_data_df) * (1 - config['data'].get('test_size', 0.2)))
        close_prices = test_data_df['close'].values[test_start_idx:]
        
        # 确保长度一致
        min_len = min(len(close_prices), len(test_sequences))
        close_prices = close_prices[:min_len]
        X_test = X_test[:min_len]
        y_test = y_test[:min_len]
        
        # 获取日期（如果有）
        test_dates = None
        if 'date' in test_data_df.columns:
            test_dates = test_data_df['date'].values[test_start_idx:test_start_idx + min_len]
    
    except Exception as e:
        print(f"数据加载出错: {str(e)}")
        return
    
    # 加载模型
    print("加载模型中...")
    model_path = "models/lstm_model.pt"
    if not os.path.exists(model_path):
        print(f"错误: 找不到训练好的模型 {model_path}")
        return
    
    try:
        # 初始化模型
        input_size = X_test.shape[2] if X_test.shape[2] > 0 else config['model']['input_size']
        model = LSTMPredictor(
            input_size=input_size, 
            hidden_size=config['model']['hidden_size']
        )
        
        # 加载模型权重
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
    except Exception as e:
        print(f"模型加载出错: {str(e)}")
        return
    
    # 允许用户自定义策略参数
    print("设置策略参数...")
    
    # 默认参数
    sma_short = 5
    sma_long = 20
    momentum_window = 10
    mr_window = 20
    max_leverage = 1.0
    
    # 尝试从用户获取参数
    try:
        print("您可以自定义策略参数，或按Enter使用默认值")
        custom_sma_short = input(f"SMA短期窗口 (默认: {sma_short}): ")
        if custom_sma_short.strip():
            sma_short = int(custom_sma_short)
        
        custom_sma_long = input(f"SMA长期窗口 (默认: {sma_long}): ")
        if custom_sma_long.strip():
            sma_long = int(custom_sma_long)
        
        custom_momentum = input(f"动量窗口 (默认: {momentum_window}): ")
        if custom_momentum.strip():
            momentum_window = int(custom_momentum)
        
        custom_mr = input(f"均值回归窗口 (默认: {mr_window}): ")
        if custom_mr.strip():
            mr_window = int(custom_mr)
        
        custom_leverage = input(f"最大杠杆 (默认: {max_leverage}): ")
        if custom_leverage.strip():
            max_leverage = float(custom_leverage)
    except:
        print("输入无效，使用默认参数")
    
    # 创建策略实例
    sma_strategy = SimpleMovingAverageCrossover(short_window=sma_short, long_window=sma_long, max_leverage=max_leverage)
    momentum_strategy = MomentumStrategy(window=momentum_window, max_leverage=max_leverage)
    mean_reversion_strategy = MeanReversionStrategy(window=mr_window, max_leverage=max_leverage)
    
    # 获取模型预测
    print("生成模型预测...")
    try:
        with torch.no_grad():
            predictions = model(X_test).squeeze().detach().cpu().numpy()
        
        true_returns = y_test.cpu().numpy()
    except Exception as e:
        print(f"模型预测出错: {str(e)}")
        return
    
    # 计算各策略回报
    print("计算策略回报...")
    model_returns = predictions * true_returns
    buyhold_returns = true_returns
    
    try:
        sma_positions = sma_strategy.generate_positions(close_prices)
        sma_returns = sma_positions * true_returns
    except Exception as e:
        print(f"SMA策略出错: {str(e)}")
        sma_returns = np.zeros_like(true_returns)
    
    try:
        momentum_positions = momentum_strategy.generate_positions(close_prices)
        momentum_returns = momentum_positions * true_returns
    except Exception as e:
        print(f"动量策略出错: {str(e)}")
        momentum_returns = np.zeros_like(true_returns)
    
    try:
        mr_positions = mean_reversion_strategy.generate_positions(close_prices)
        mr_returns = mr_positions * true_returns
    except Exception as e:
        print(f"均值回归策略出错: {str(e)}")
        mr_returns = np.zeros_like(true_returns)
    
    # 计算累积回报
    print("计算累积回报...")
    cumulative_model = np.cumprod(1 + model_returns) - 1
    cumulative_buyhold = np.cumprod(1 + buyhold_returns) - 1
    cumulative_sma = np.cumprod(1 + sma_returns) - 1
    cumulative_momentum = np.cumprod(1 + momentum_returns) - 1
    cumulative_mr = np.cumprod(1 + mr_returns) - 1
    
    # 计算性能指标
    def calc_metrics(returns):
        # 防止数值溢出，确保夏普比率等指标在合理范围内
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8
        sharpe = mean_return / std_return * np.sqrt(252)
        sharpe = max(min(sharpe, 10), -10)  # 限制在合理范围内
        
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
    mr_metrics = calc_metrics(mr_returns)
    
    # 绘制累积回报比较图
    print("绘制比较图...")
    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_model, label=f'LSTM模型 (Sharpe: {model_metrics["sharpe"]:.2f})')
    plt.plot(cumulative_buyhold, label=f'买入持有 (Sharpe: {buyhold_metrics["sharpe"]:.2f})')
    plt.plot(cumulative_sma, label=f'{sma_strategy.name} (Sharpe: {sma_metrics["sharpe"]:.2f})')
    plt.plot(cumulative_momentum, label=f'{momentum_strategy.name} (Sharpe: {momentum_metrics["sharpe"]:.2f})')
    plt.plot(cumulative_mr, label=f'{mean_reversion_strategy.name} (Sharpe: {mr_metrics["sharpe"]:.2f})')
    
    plt.title('策略比较: 累积回报')
    plt.xlabel('时间')
    plt.ylabel('累积回报')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if test_dates is not None:
        n_ticks = min(10, len(test_dates))
        tick_indices = np.linspace(0, len(test_dates)-1, n_ticks, dtype=int)
        plt.xticks(tick_indices, [test_dates[i] for i in tick_indices], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 创建性能指标比较表格
    metrics_df = pd.DataFrame({
        'LSTM模型': [model_metrics['sharpe'], model_metrics['max_drawdown'], model_metrics['win_rate'], model_metrics['final_return']],
        '买入持有': [buyhold_metrics['sharpe'], buyhold_metrics['max_drawdown'], buyhold_metrics['win_rate'], buyhold_metrics['final_return']],
        sma_strategy.name: [sma_metrics['sharpe'], sma_metrics['max_drawdown'], sma_metrics['win_rate'], sma_metrics['final_return']],
        momentum_strategy.name: [momentum_metrics['sharpe'], momentum_metrics['max_drawdown'], momentum_metrics['win_rate'], momentum_metrics['final_return']],
        mean_reversion_strategy.name: [mr_metrics['sharpe'], mr_metrics['max_drawdown'], mr_metrics['win_rate'], mr_metrics['final_return']]
    }, index=['夏普比率', '最大回撤', '胜率', '最终回报'])
    
    display(metrics_df.style.format("{:.4f}")
            .highlight_max(subset=['夏普比率', '胜率', '最终回报'], axis=1, color='lightgreen')
            .highlight_min(subset=['最大回撤'], axis=1, color='lightgreen')
            .highlight_min(subset=['夏普比率', '胜率', '最终回报'], axis=1, color='lightcoral')
            .highlight_max(subset=['最大回撤'], axis=1, color='lightcoral'))
    
    # 统计显著性测试
    try:
        t_stat, p_value = stats.ttest_ind(model_returns, buyhold_returns, equal_var=False)
        print(f"\n统计显著性测试 (LSTM vs 买入并持有):")
        print(f"t值: {t_stat:.4f}, p值: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
        print(f"在95%置信水平下LSTM模型{'优于' if p_value < 0.05 and t_stat > 0 else '不优于'}买入并持有策略")
    except Exception as e:
        print(f"统计显著性测试出错: {str(e)}")
    
    return {
        'model_metrics': model_metrics,
        'buyhold_metrics': buyhold_metrics,
        'sma_metrics': sma_metrics,
        'momentum_metrics': momentum_metrics,
        'mr_metrics': mr_metrics
    }

# 运行比较函数
# run_baseline_comparison()  # 取消注释此行运行比较 