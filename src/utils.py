import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os
import json


def positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """正弦位置编码"""
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding = torch.zeros(max_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)

    return pos_encoding.unsqueeze(0)


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """创建填充掩码 - 修复维度问题"""
    # 形状: (batch_size, 1, 1, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(seq_len: int) -> torch.Tensor:
    """创建前瞻掩码 - 修复维度问题"""
    # 形状: (1, 1, seq_len, seq_len)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(1) == 1  # (1, 1, seq_len, seq_len)

def get_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    """计算准确率（排除填充标记）"""
    mask = targets != pad_idx
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets) & mask
    return correct.float().sum().item() / mask.float().sum().item()


def save_training_curves(train_losses: List[float], val_losses: List[float],
                         train_accs: List[float], val_accs: List[float],
                         save_path: str):
    """保存训练曲线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 损失图
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率图
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class CheckpointManager:
    """模型检查点管理"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, model: nn.Module, optimizer, scheduler,
                        epoch: int, loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss
        }

        filename = f'checkpoint_epoch_{epoch}.pt'
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, model: nn.Module, optimizer=None, scheduler=None,
                        checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pt')

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint['loss']