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
        
        # åŠ¨æ€å‡¯åˆ©ç³»æ•°è®¡ç®—
        kelly_f = torch.clamp(mu / (sigma**2 + 1e-6), 0.0, self.max_leverage)
        
        # ç»„åˆæŸå¤±é¡¹
        position_loss = torch.mean((preds - kelly_f)**2)
        return_penalty = -mu  # è´Ÿå·è¡¨ç¤ºæœ€å¤§åŒ–æ”¶ç›Š
        
        return self.alpha * position_loss + (1 - self.alpha) * return_penalty
