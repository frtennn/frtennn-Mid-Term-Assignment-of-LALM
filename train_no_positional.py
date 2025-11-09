import torch
import argparse
import os
import sys
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import math
import time

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import load_config, ModelConfig, TrainingConfig
from model import Transformer
from data_loader import get_data_loader
from utils import create_padding_mask, create_look_ahead_mask, get_accuracy, CheckpointManager
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn


def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NoPositionalTransformer(Transformer):
    """无位置编码的Transformer变体"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 max_seq_length: int, d_model: int = 512, n_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__(src_vocab_size, tgt_vocab_size, max_seq_length,
                         d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         d_ff, dropout)

        # 将位置编码置零
        zero_pos_encoding = torch.zeros_like(self.encoder.pos_encoding)
        self.encoder.pos_encoding = zero_pos_encoding
        self.decoder.pos_encoding = zero_pos_encoding


class NoPositionalTrainer:
    """无位置编码模型训练器"""

    def __init__(self, model_config, training_config, device):
        self.model_config = model_config
        self.training_config = training_config
        self.device = device

        # 创建保存目录
        self.save_dir = 'no_positional_model'
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化模型
        self.model = NoPositionalTransformer(
            src_vocab_size=model_config.vocab_size,
            tgt_vocab_size=model_config.vocab_size,
            max_seq_length=model_config.max_seq_length,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            num_encoder_layers=model_config.num_encoder_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            d_ff=model_config.d_ff,
            dropout=model_config.dropout
        ).to(device)

        # 打印模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"无位置编码模型参数总数: {total_params:,}")

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            betas=training_config.betas,
            eps=training_config.eps,
            weight_decay=training_config.weight_decay
        )

        # 学习率调度器
        self.scheduler = self._get_scheduler()

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=training_config.label_smoothing
        )

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(self.save_dir)

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def _get_scheduler(self):
        """Transformer学习率调度器"""

        def lr_lambda(step):
            step = max(step, 1)
            return min(
                step ** -0.5,
                step * (self.training_config.warmup_steps ** -1.5)
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为Transformer创建掩码"""
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()

        # 源序列填充掩码
        src_mask = create_padding_mask(src, pad_idx=0)

        # 目标序列填充掩码
        tgt_padding_mask = create_padding_mask(tgt, pad_idx=0)

        # 目标序列前瞻掩码
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(self.device)

        # 组合目标掩码
        tgt_padding_mask_expanded = tgt_padding_mask.expand(-1, -1, tgt_len, -1)
        tgt_mask = tgt_padding_mask_expanded & tgt_look_ahead_mask

        # 编码器-解码器注意力掩码
        memory_mask = create_padding_mask(src, pad_idx=0)

        return src_mask, tgt_mask, memory_mask

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个周期"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training No Positional Encoding")

        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 准备教师强制数据
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 创建掩码
            src_mask, tgt_mask, memory_mask = self.create_masks(src, tgt_input)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask, memory_mask)

            # 计算损失
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_output.contiguous().view(-1)
            )

            # 反向传播和梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            # 计算准确率
            acc = get_accuracy(output, tgt_output, pad_idx=0)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{acc:.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        return total_loss / num_batches, total_acc / num_batches

    def validate(self, val_loader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc="Validating No Positional Encoding"):
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask, tgt_mask, memory_mask = self.create_masks(src, tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask, memory_mask)

                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_output.contiguous().view(-1)
                )

                acc = get_accuracy(output, tgt_output, pad_idx=0)

                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def train(self, num_epochs: int = 5):
        """训练模型"""
        print("开始训练无位置编码模型...")

        # 数据加载器
        train_loader = get_data_loader(
            self.training_config.batch_size,
            self.model_config.max_seq_length,
            'train',
            self.model_config.vocab_size,
            rebuild_vocab=False
        )
        val_loader = get_data_loader(
            self.training_config.batch_size,
            self.model_config.max_seq_length,
            'validation',
            self.model_config.vocab_size,
            rebuild_vocab=False
        )

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nNo Positional Encoding - Epoch {epoch + 1}/{num_epochs}")

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 保存训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"No Positional Encoding - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"No Positional Encoding - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            # 保存检查点
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, val_loss, is_best
            )

        print("No Positional Encoding training completed!")
        return self.train_losses, self.val_losses, self.train_accs, self.val_accs


def create_eval_masks(src: torch.Tensor, tgt_input: torch.Tensor, device: torch.device):
    """为评估创建掩码"""
    batch_size, src_len = src.size()
    _, tgt_len = tgt_input.size()

    # 源序列填充掩码
    src_mask = create_padding_mask(src, pad_idx=0)

    # 目标序列填充掩码
    tgt_padding_mask = create_padding_mask(tgt_input, pad_idx=0)

    # 目标序列前瞻掩码
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(device)

    # 组合目标掩码
    tgt_padding_mask_expanded = tgt_padding_mask.expand(-1, -1, tgt_len, -1)
    tgt_mask = tgt_padding_mask_expanded & tgt_look_ahead_mask

    # 编码器-解码器注意力掩码
    memory_mask = create_padding_mask(src, pad_idx=0)

    return src_mask, tgt_mask, memory_mask


def evaluate_no_positional_model(model_config: ModelConfig, training_config, device: torch.device, seed: int):
    """评估无位置编码模型"""
    set_seed(seed)

    model_path = "no_positional_model/best_model.pt"

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        return None

    # 初始化模型
    model = NoPositionalTransformer(
        src_vocab_size=model_config.vocab_size,
        tgt_vocab_size=model_config.vocab_size,
        max_seq_length=model_config.max_seq_length,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        num_encoder_layers=model_config.num_encoder_layers,
        num_decoder_layers=model_config.num_decoder_layers,
        d_ff=model_config.d_ff,
        dropout=model_config.dropout
    ).to(device)

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"Loaded no positional encoding model from epoch {checkpoint['epoch']}, validation loss: {checkpoint['loss']:.4f}")

    # 加载测试数据
    try:
        test_loader = get_data_loader(
            training_config.batch_size,
            model_config.max_seq_length,
            'test',
            model_config.vocab_size,
            rebuild_vocab=False
        )
    except:
        print("Test set not available, using validation set for evaluation")
        test_loader = get_data_loader(
            training_config.batch_size,
            model_config.max_seq_length,
            'validation',
            model_config.vocab_size,
            rebuild_vocab=False
        )

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)

    # 评估模型
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Testing No Positional Encoding"):
            src, tgt = src.to(device), tgt.to(device)

            # 准备教师强制数据
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 创建掩码
            src_mask, tgt_mask, memory_mask = create_eval_masks(src, tgt_input, device)

            # 前向传播
            output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)

            # 计算损失
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_output.contiguous().view(-1)
            )

            # 计算准确率
            acc = get_accuracy(output, tgt_output, pad_idx=0)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

    test_loss = total_loss / num_batches
    test_acc = total_acc / num_batches

    results = {
        'No Positional Encoding': {
            'Test Loss': test_loss,
            'Test Accuracy': test_acc,
            'Parameters': sum(p.numel() for p in model.parameters())
        }
    }

    print(f"No Positional Encoding - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return results


def plot_training_curves(trainer, save_path: str):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(trainer.train_losses) + 1)

    # 损失图
    ax1.plot(epochs, trainer.train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, trainer.val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('No Positional Encoding - Training Curves')
    ax1.legend()
    ax1.grid(True)

    # 准确率图
    ax2.plot(epochs, trainer.train_accs, label='Train Accuracy', marker='o')
    ax2.plot(epochs, trainer.val_accs, label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('No Positional Encoding - Accuracy Curves')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate no positional encoding model')
    parser.add_argument('--config', type=str, default='configs/ablation_8.yaml', help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate, do not train')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    model_config, training_config = load_config(args.config)

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    if not args.evaluate_only:
        # 训练无位置编码模型
        set_seed(args.seed)
        trainer = NoPositionalTrainer(model_config, training_config, device)
        train_losses, val_losses, train_accs, val_accs = trainer.train(num_epochs=args.epochs)

        # 绘制训练曲线
        plot_training_curves(trainer, "results/no_positional_training_curves.png")

    # 评估无位置编码模型
    print("\n开始评估无位置编码模型...")
    results = evaluate_no_positional_model(model_config, training_config, device, args.seed)

    if results:
        # 保存结果
        with open("results/no_positional_evaluation.txt", 'w', encoding='utf-8') as f:
            f.write("No Positional Encoding Model Evaluation\n")
            f.write("=" * 50 + "\n")
            for variant, metrics in results.items():
                f.write(
                    f"{variant}: Test Loss = {metrics['Test Loss']:.4f}, Test Accuracy = {metrics['Test Accuracy']:.4f}\n")

        print(f"\n无位置编码模型评估结果:")
        print(f"测试损失: {results['No Positional Encoding']['Test Loss']:.4f}")
        print(f"测试准确率: {results['No Positional Encoding']['Test Accuracy']:.4f}")
        print(f"结果已保存至: results/no_positional_evaluation.txt")


if __name__ == "__main__":
    main()