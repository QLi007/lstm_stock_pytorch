import torch
import torch.nn as nn
import numpy as np

class TradingOptimizationLoss(nn.Module):
    """
    Advanced loss function for trading strategy optimization.
    
    Optimizes for:
    1. Maximizing returns (Sharpe ratio)
    2. Minimizing drawdowns
    3. Maintaining smooth equity curve (low volatility)
    """
    def __init__(self, return_weight=1.0, drawdown_weight=1.0, 
                 smoothness_weight=0.5, max_leverage=2.0):
        """
        Args:
            return_weight: Weight for the returns component
            drawdown_weight: Weight for the drawdown component
            smoothness_weight: Weight for the smoothness (low volatility) component
            max_leverage: Maximum allowed leverage
        """
        super().__init__()
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.smoothness_weight = smoothness_weight
        self.max_leverage = max_leverage
    
    def forward(self, positions, returns):
        """
        Calculate loss based on positions and actual returns.
        
        Args:
            positions: Model outputs (position sizes) - range [-1, 1]
            returns: Actual future returns
            
        Returns:
            Loss value to minimize
        """
        # Clamp positions to max leverage
        positions = torch.clamp(positions, -self.max_leverage, self.max_leverage)
        
        # Calculate portfolio returns
        portfolio_returns = positions * returns
        
        # 1. Return component (negative sharpe ratio - we want to maximize it)
        mean_return = torch.mean(portfolio_returns)
        return_std = torch.std(portfolio_returns) + 1e-6  # Avoid division by zero
        sharpe = mean_return / return_std
        return_loss = -sharpe  # Negative because we want to maximize Sharpe
        
        # 2. Drawdown component
        # Calculate cumulative returns
        cumulative_returns = torch.cumsum(portfolio_returns, dim=0)
        
        # Calculate running maximum of cumulative returns
        running_max = torch.zeros_like(cumulative_returns)
        current_max = torch.tensor(0.0, device=positions.device)
        
        for i in range(len(cumulative_returns)):
            current_max = torch.maximum(current_max, cumulative_returns[i])
            running_max[i] = current_max
        
        # Calculate drawdowns
        drawdowns = running_max - cumulative_returns
        max_drawdown = torch.max(drawdowns)
        avg_drawdown = torch.mean(drawdowns)
        
        # Drawdown loss - penalize large drawdowns
        drawdown_loss = max_drawdown + 0.5 * avg_drawdown
        
        # 3. Smoothness component (volatility of returns)
        # Calculate return differences (day-to-day changes)
        return_diff = torch.diff(portfolio_returns, dim=0)
        return_volatility = torch.std(return_diff)
        
        # Smoothness loss - penalize high volatility
        smoothness_loss = return_volatility
        
        # 4. Position sizing regularization (optional) - penalize rapid position changes
        position_diff = torch.diff(positions, dim=0)
        position_volatility = torch.mean(torch.abs(position_diff))
        
        # Combined loss
        total_loss = (self.return_weight * return_loss + 
                      self.drawdown_weight * drawdown_loss +
                      self.smoothness_weight * smoothness_loss +
                      0.2 * position_volatility)
        
        return total_loss


class KellyDrawdownLoss(nn.Module):
    """
    Combined Kelly Criterion and Drawdown optimization.
    
    Combines:
    1. Kelly Criterion for position sizing
    2. Drawdown penalty
    3. Smoothness penalty for equity curve
    """
    def __init__(self, alpha=0.5, max_leverage=2.0, dd_weight=1.0, smoothness_weight=0.5):
        super().__init__()
        self.alpha = alpha  # Weight between Kelly position and return maximization
        self.max_leverage = max_leverage
        self.dd_weight = dd_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, preds, targets):
        # Portfolio returns based on predicted positions and actual returns
        portfolio_returns = preds * targets
        
        # Kelly criterion component
        mu = torch.mean(portfolio_returns)
        sigma = torch.std(portfolio_returns) + 1e-6
        
        # Optimal Kelly fraction
        kelly_f = torch.clamp(mu / (sigma**2 + 1e-6), -self.max_leverage, self.max_leverage)
        
        # Position sizing loss (MSE to optimal Kelly fraction)
        position_loss = torch.mean((preds - kelly_f)**2)
        
        # Return maximization (negative mean return)
        return_loss = -mu
        
        # Drawdown calculation
        cum_returns = torch.cumsum(portfolio_returns, dim=0)
        running_max = torch.cummax(cum_returns, dim=0)[0]
        drawdowns = running_max - cum_returns
        max_drawdown = torch.max(drawdowns)
        
        # Smoothness calculation - penalize large swings in returns
        if len(portfolio_returns) > 1:
            return_changes = torch.diff(portfolio_returns, dim=0)
            smoothness_loss = torch.std(return_changes)
        else:
            smoothness_loss = torch.tensor(0.0, device=preds.device)
        
        # Combine losses with weights
        kelly_component = self.alpha * position_loss + (1 - self.alpha) * return_loss
        drawdown_component = max_drawdown * self.dd_weight
        smoothness_component = smoothness_loss * self.smoothness_weight
        
        total_loss = kelly_component + drawdown_component + smoothness_component
        
        return total_loss 