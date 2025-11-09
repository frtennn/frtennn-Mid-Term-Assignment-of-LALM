import argparse
import torch
import random
import numpy as np
from src.config import load_config
from src.train import TransformerTrainer


def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Transformer从零开始训练')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')

    args = parser.parse_args()

    # 设置种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    model_config, training_config = load_config(args.config)

    # 初始化和运行训练器
    trainer = TransformerTrainer(model_config, training_config, device)
    trainer.train()


if __name__ == "__main__":
    main()