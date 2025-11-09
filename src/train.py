import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import os
import time
from typing import Tuple

from .model import Transformer
from .data_loader import get_data_loader
from .utils import (create_padding_mask, create_look_ahead_mask,
                    get_accuracy, save_training_curves, CheckpointManager)


class TransformerTrainer:
    """Transformer模型训练器，包含所有训练稳定性技巧"""

    def __init__(self, model_config, training_config, device):
        self.model_config = model_config
        self.training_config = training_config
        self.device = device

        # 初始化模型
        self.model = Transformer(
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

        # 打印模型统计信息
        self._print_model_stats()

        # 使用AdamW初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            betas=training_config.betas,
            eps=training_config.eps,
            weight_decay=training_config.weight_decay
        )

        # 学习率调度器（Transformer风格）
        self.scheduler = self._get_scheduler()

        # 带标签平滑的损失函数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # 填充标记
            label_smoothing=training_config.label_smoothing
        )

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(training_config.save_dir)

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def _print_model_stats(self):
        """打印模型参数统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"模型参数:")
        print(f"  总数: {total_params:,}")
        print(f"  可训练: {trainable_params:,}")
        print(f"  模型配置:")
        for key, value in self.model_config.__dict__.items():
            print(f"    {key}: {value}")

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
        """为Transformer创建掩码 - 修复维度问题"""
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()

        # 源序列填充掩码 (batch_size, 1, 1, src_len)
        src_mask = create_padding_mask(src, pad_idx=0)

        # 目标序列填充掩码 (batch_size, 1, 1, tgt_len)
        tgt_padding_mask = create_padding_mask(tgt, pad_idx=0)

        # 目标序列前瞻掩码 (1, 1, tgt_len, tgt_len)
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(self.device)

        # 组合目标掩码 (batch_size, 1, tgt_len, tgt_len)
        tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

        # 编码器-解码器注意力掩码 (batch_size, 1, 1, src_len)
        memory_mask = create_padding_mask(src, pad_idx=0)

        return src_mask, tgt_mask, memory_mask

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个周期 - 添加调试信息"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="训练中")

        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 调试信息
            if batch_idx == 0:
                print(f"调试 - 批次 {batch_idx}:")
                print(f"  src形状: {src.shape}")
                print(f"  tgt形状: {tgt.shape}")

            # 准备教师强制数据
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 创建掩码
            src_mask, tgt_mask, memory_mask = self.create_masks(src, tgt_input)

            # 调试信息
            if batch_idx == 0:
                print(f"  src_mask形状: {src_mask.shape}")
                print(f"  tgt_mask形状: {tgt_mask.shape}")
                print(f"  memory_mask形状: {memory_mask.shape}")
                print(f"  tgt_input形状: {tgt_input.shape}")
                print(f"  tgt_output形状: {tgt_output.shape}")

            # 前向传播
            self.optimizer.zero_grad()
            try:
                output = self.model(src, tgt_input, src_mask, tgt_mask, memory_mask)

                # 调试信息
                if batch_idx == 0:
                    print(f"  output形状: {output.shape}")
                    print(
                        f"  预期输出形状: ({tgt_output.shape[0]}, {tgt_output.shape[1]}, {self.model_config.vocab_size})")

            except Exception as e:
                print(f"前向传播错误: {e}")
                print(f"src: {src.shape}, tgt_input: {tgt_input.shape}")
                print(f"src_mask: {src_mask.shape}, tgt_mask: {tgt_mask.shape}")
                raise e

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
                '损失': f'{loss.item():.4f}',
                '准确率': f'{acc:.4f}',
                '学习率': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        return total_loss / num_batches, total_acc / num_batches

    def validate(self, val_loader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc="验证中"):
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

    def train(self):
        """完整的训练循环"""
        print("开始训练...")

        # 数据加载器
        train_loader = get_data_loader(
            self.training_config.batch_size,
            self.model_config.max_seq_length,
            'train',
            self.model_config.vocab_size,
            rebuild_vocab=True  # 第一次运行时重建词汇表
        )
        val_loader = get_data_loader(
            self.training_config.batch_size,
            self.model_config.max_seq_length,
            'validation',
            self.model_config.vocab_size,
            rebuild_vocab=False
        )

        best_val_loss = float('inf')

        for epoch in range(self.training_config.num_epochs):
            print(f"\n第 {epoch + 1}/{self.training_config.num_epochs} 轮")

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 保存训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

            # 保存检查点
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, val_loss, is_best
            )

            # 每5轮保存一次训练曲线
            if (epoch + 1) % 5 == 0:
                save_training_curves(
                    self.train_losses, self.val_losses,
                    self.train_accs, self.val_accs,
                    f"results/training_curves_epoch_{epoch + 1}.png"
                )

        print("训练完成!")

        # 保存最终训练曲线
        save_training_curves(
            self.train_losses, self.val_losses,
            self.train_accs, self.val_accs,
            "results/final_training_curves.png"
        )


def main():
    # 配置
    from .config import load_config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    model_config, training_config = load_config('configs/base.yaml')

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    # 初始化训练器
    trainer = TransformerTrainer(model_config, training_config, device)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()