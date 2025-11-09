import torch
import argparse
import os
import sys
import random
import numpy as np
from tqdm import tqdm
import sacrebleu
from typing import List, Tuple
import math

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import load_config
from train_no_positional import NoPositionalTransformer
from data_loader import get_data_loader
from utils import create_padding_mask, create_look_ahead_mask, get_accuracy
import sentencepiece as spm


def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_masks(src: torch.Tensor, tgt: torch.Tensor, device: torch.device) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """为无位置编码Transformer创建掩码"""
    batch_size, src_len = src.size()
    _, tgt_len = tgt.size()

    # 源序列填充掩码
    src_mask = create_padding_mask(src, pad_idx=0)

    # 目标序列填充掩码和前瞻掩码
    tgt_padding_mask = create_padding_mask(tgt, pad_idx=0)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(device)

    # 组合目标掩码
    tgt_padding_mask_expanded = tgt_padding_mask.expand(-1, -1, tgt_len, -1)
    tgt_mask = tgt_padding_mask_expanded & tgt_look_ahead_mask

    # 编码器-解码器注意力掩码
    memory_mask = create_padding_mask(src, pad_idx=0)

    return src_mask, tgt_mask, memory_mask


def evaluate_model(model: NoPositionalTransformer, test_loader, device: torch.device, criterion):
    """评估无位置编码模型在测试集上的表现"""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    total_tokens = 0  # 用于计算困惑度的总token数

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="测试无位置编码模型"):
            src, tgt = src.to(device), tgt.to(device)

            # 准备教师强制数据
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 创建掩码
            src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input, device)

            # 前向传播（无位置编码）
            output = model(src, tgt_input, src_mask, tgt_mask, memory_mask, use_pos_encoding=False)

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

            # 计算非填充token的数量用于困惑度计算
            non_pad_tokens = (tgt_output != 0).sum().item()
            total_tokens += non_pad_tokens

            # 收集预测结果用于BLEU计算
            predictions = output.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(tgt_output.cpu().numpy())

    # 计算平均损失和困惑度
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)  # 批平均困惑度

    return avg_loss, total_acc / num_batches, all_predictions, all_references, perplexity


def decode_tokens(tokens, sp_processor):
    """安全地解码tokens为文本"""
    try:
        # 方法1: 使用DecodeIds
        return sp_processor.DecodeIds([int(t) for t in tokens])
    except:
        try:
            # 方法2: 使用decode_pieces
            pieces = [sp_processor.IdToPiece(int(t)) for t in tokens]
            return sp_processor.DecodePieces(pieces)
        except:
            # 方法3: 手动处理
            text = ""
            for token in tokens:
                token = int(token)
                if token == 0 or token == sp_processor.eos_id() or token == sp_processor.bos_id():
                    continue
                piece = sp_processor.IdToPiece(token)
                if piece.startswith("▁"):
                    text += " " + piece[1:]
                else:
                    text += piece
            return text.strip()


def calculate_bleu(predictions: List[List[int]], references: List[List[int]], sp_processor):
    """计算BLEU分数"""
    # 将索引转换为文本
    pred_texts = []
    ref_texts = []

    for pred, ref in zip(predictions, references):
        # 解码预测文本（去掉填充和特殊标记）
        pred_tokens = []
        for token in pred:
            if token == 0:  # 填充标记
                continue
            if token == sp_processor.eos_id() or token == sp_processor.bos_id():
                continue
            pred_tokens.append(token)

        # 解码参考文本
        ref_tokens = []
        for token in ref:
            if token == 0:  # 填充标记
                continue
            if token == sp_processor.eos_id() or token == sp_processor.bos_id():
                continue
            ref_tokens.append(token)

        if pred_tokens and ref_tokens:
            try:
                pred_text = decode_tokens(pred_tokens, sp_processor)
                ref_text = decode_tokens(ref_tokens, sp_processor)

                # 确保文本不为空
                if pred_text.strip() and ref_text.strip():
                    pred_texts.append(pred_text)
                    ref_texts.append([ref_text])  # sacrebleu需要引用列表
            except Exception as e:
                print(f"解码错误: {e}")
                continue

    # 计算BLEU分数
    if pred_texts and ref_texts:
        try:
            bleu = sacrebleu.corpus_bleu(pred_texts, ref_texts)
            return bleu.score, pred_texts, ref_texts
        except Exception as e:
            print(f"BLEU计算错误: {e}")
            return 0.0, [], []
    else:
        return 0.0, [], []


def show_example(model: NoPositionalTransformer, test_loader, device: torch.device, sp_processor,
                 example_index: int = 0):
    """显示测试集中的一个例子及其翻译结果"""
    model.eval()

    # 获取测试集中的一个批次
    for i, (src_batch, tgt_batch) in enumerate(test_loader):
        if i > 0:  # 只取第一个批次
            break

    if example_index >= len(src_batch):
        example_index = 0

    src = src_batch[example_index:example_index + 1].to(device)
    tgt = tgt_batch[example_index:example_index + 1].to(device)

    # 解码源文本
    src_tokens = src[0].cpu().numpy()
    src_text = decode_tokens([t for t in src_tokens if t not in [0, sp_processor.bos_id(), sp_processor.eos_id()]],
                             sp_processor)

    # 解码参考目标文本
    tgt_tokens = tgt[0].cpu().numpy()
    ref_text = decode_tokens([t for t in tgt_tokens if t not in [0, sp_processor.bos_id(), sp_processor.eos_id()]],
                             sp_processor)

    print("\n" + "=" * 50)
    print("无位置编码模型测试样例展示:")
    print("=" * 50)
    print(f"源文本 (英语): {src_text}")
    print(f"参考翻译 (德语): {ref_text}")

    # 简单显示模型输出（使用教师强制）
    with torch.no_grad():
        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input, device)
        output = model(src, tgt_input, src_mask, tgt_mask, memory_mask, use_pos_encoding=False)

        # 获取预测的token
        pred_tokens = output.argmax(dim=-1)[0].cpu().numpy()
        pred_text = decode_tokens(
            [t for t in pred_tokens if t not in [0, sp_processor.bos_id(), sp_processor.eos_id()]], sp_processor)
        print(f"模型输出 (德语): {pred_text}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='测试无位置编码Transformer模型')
    parser.add_argument('--config', type=str, default='configs/ablation_8.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='no_positional_model/best_model.pt',
                        help='无位置编码模型检查点文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"测试模型: 无位置编码Transformer")

    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点文件 {args.checkpoint} 不存在!")
        print("请先运行 train_no_positional.py 训练模型")
        return

    # 加载配置
    model_config, training_config = load_config(args.config)

    # 初始化无位置编码模型
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载无位置编码模型检查点，来自第 {checkpoint['epoch']} 轮，验证损失: {checkpoint['loss']:.4f}")

    # 加载测试数据
    try:
        test_loader = get_data_loader(
            batch_size=training_config.batch_size,
            max_length=model_config.max_seq_length,
            split='test',
            vocab_size=model_config.vocab_size,
            rebuild_vocab=False
        )
    except:
        print("测试集不可用，使用验证集进行评估")
        test_loader = get_data_loader(
            batch_size=training_config.batch_size,
            max_length=model_config.max_seq_length,
            split='validation',
            vocab_size=model_config.vocab_size,
            rebuild_vocab=False
        )

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)

    # 加载分词器用于BLEU计算
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.Load('vocab.model')

    # 评估模型
    print("开始在测试集上评估无位置编码模型...")
    test_loss, test_acc, predictions, references, perplexity = evaluate_model(model, test_loader, device, criterion)

    # 计算BLEU分数
    bleu_score, pred_texts, ref_texts = calculate_bleu(predictions, references, sp_processor)

    print("\n" + "=" * 60)
    print("无位置编码模型测试结果:")
    print("=" * 60)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"困惑度: {perplexity:.2f}")
    print(f"BLEU分数: {bleu_score:.2f}")
    print("=" * 60)

    # 显示一个例子
    show_example(model, test_loader, device, sp_processor, example_index=args.seed % 10)

    # 保存结果到文件
    result_file = "no_positional_model_test_results.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("无位置编码Transformer模型测试结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试损失: {test_loss:.4f}\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"困惑度: {perplexity:.2f}\n")
        f.write(f"BLEU分数: {bleu_score:.2f}\n")
        f.write(f"模型参数: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"配置文件: {args.config}\n")

    print(f"\n详细结果已保存至: {result_file}")


if __name__ == "__main__":
    main()