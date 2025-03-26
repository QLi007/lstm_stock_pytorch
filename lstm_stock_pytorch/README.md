# LSTM Stock Predictor with Kelly Criterion

åŸºäºŽPyTorchçš„é‡åŒ–äº¤æ˜"ç³»ç»Ÿï¼Œé›†æˆåŠ¨æ€å‡¯åˆ©å…¬å¼ä»"ä½ç®¡ç0†

## åŠŸèƒ½ç‰¹æ€§
- æ—¶é—´åºåˆ—å®‰å…¨å¤„ç0†ï¼ˆä¸æ ¼é¿å…æœªæå‡½æ•°ï¼‰
- LSTMä»·æ ¼é¢„æµ‹æ¨¡åž‹
- åŠ¨æ€é£Žé™©æŽ§åˆ¶ç–ç•¥
- èªåŠ¨åŒ–CI/CDæµæ°´çº¿
- å•å…æµ‹è¯•è¦†ç›–æ ¸å¿ƒåŠŸèƒ½

## å¿«é€Ÿå¼€å§‹
`ash
# å®‰è£…ä¾èµ–
pip install -r requirements.txt

# è¿è¡Œè®­ç»ƒ
python src/train.py

# æ‰è¡Œæµ‹è¯•
python -m pytest tests/ -v
`

## 模型评估与可视化

```bash
# 训练模型并保存
python src/train.py --save-model

# 评估模型性能
python src/evaluate.py --visualize

# 使用自定义参数
python src/evaluate.py --model-path models/custom_model.pt --save-plot results/performance.png
```

## 项目结构

```
.
├── configs/             # 配置文件
│   └── default.yaml     # 默认参数配置
├── data/                # 数据文件
│   └── sample.csv       # 示例股票数据
├── src/                 # 源代码
│   ├── data/            # 数据处理
│   │   └── loader.py    # 数据加载器
│   ├── model/           # 模型定义
│   │   ├── lstm.py      # LSTM模型
│   │   └── kelly_loss.py# 凯利准则损失函数
│   ├── train.py         # 训练脚本
│   └── evaluate.py      # 评估脚本
├── tests/               # 单元测试
└── models/              # 保存的模型（自动创建）
```

