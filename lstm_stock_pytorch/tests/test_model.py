import torch
import pytest
from src.model.lstm import LSTMPredictor
from src.model.kelly_loss import KellyLoss

def test_lstm_forward_pass():
    model = LSTMPredictor(input_size=4)
    x = torch.randn(32, 30, 4)  # (batch, seq_len, features)
    out = model(x)
    assert out.shape == (32, 1), "è¾“å‡ºå½¢çŠ¶åº”ä¸º(batch_size, 1)"
    assert torch.all(out >= -1) and torch.all(out <= 1), "è¾“å‡ºåº”åœ¨[-1, 1]èŒƒå›´å†…"

def test_kelly_loss_calculation():
    criterion = KellyLoss(max_leverage=2.0)
    preds = torch.tensor([0.8, 1.2, -0.5], dtype=torch.float32)
    targets = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float32)
    loss = criterion(preds, targets)
    assert loss.item() > 0, "æŸå¤±å€¼åº”ä¸ºæ­£æ•°"
    assert not torch.isnan(loss), "æŸå¤±å€¼ä¸åº”ä¸ºNaN"
