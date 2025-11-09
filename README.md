# Transformer从零实现

一个完整的Transformer模型从零实现，用于IWSLT2017英德数据集的机器翻译任务。

## 要求

- Python 3.8+
- PyTorch 2.4.0
- 推荐使用支持CUDA的GPU

## 安装

```bash
pip install -r requirements.txt
```

## 目录
transformer-from-scratch/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── data_loader.py
│   ├── config.py
│   └── utils.py
├── configs/
│   └── base.yaml
├── scripts/
│   └── run.sh
├── requirements.txt
├── README.md
└── train.py
└── test.py
└── train_no_positional.py
└── test_no_positional.py