import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List
import sentencepiece as spm
from datasets import load_dataset

import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

class IWSLT2017Dataset(Dataset):
    """IWSLT2017英德数据集加载器"""

    def __init__(self, split: str = 'train', max_length: int = 256,
                 vocab_size: int = 30000, rebuild_vocab: bool = False):
        self.split = split
        self.max_length = max_length

        # 加载数据集
        self.dataset = load_dataset("iwslt2017", "iwslt2017-de-en", split=split)

        # 构建词汇表
        if rebuild_vocab or not os.path.exists('vocab.model'):
            self._build_vocabulary(vocab_size)

        # 加载分词器
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('vocab.model')

        # 预处理数据
        self.data = self._preprocess_data()

    def _build_vocabulary(self, vocab_size: int):
        """使用SentencePiece构建词汇表"""
        print("构建词汇表...")

        # 保存训练文本
        with open('train_text.txt', 'w', encoding='utf-8') as f:
            for item in self.dataset:
                f.write(f"{item['translation']['en']}\n")
                f.write(f"{item['translation']['de']}\n")

        # 训练SentencePiece模型
        spm.SentencePieceTrainer.Train(
            f'--input=train_text.txt --model_prefix=vocab '
            f'--vocab_size={vocab_size} --character_coverage=1.0 '
            f'--model_type=bpe --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
        )

        print("词汇表构建完成")

    def _preprocess_data(self) -> List[Tuple[str, str]]:
        """预处理数据"""
        data = []
        for item in self.dataset:
            en_text = item['translation']['en']
            de_text = item['translation']['de']
            data.append((en_text, de_text))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        en_text, de_text = self.data[idx]

        # 分词并转换为索引
        src_tokens = self.sp.EncodeAsIds(en_text)
        tgt_tokens = self.sp.EncodeAsIds(de_text)

        # 添加BOS和EOS标记
        src_tokens = [self.sp.bos_id()] + src_tokens + [self.sp.eos_id()]
        tgt_tokens = [self.sp.bos_id()] + tgt_tokens + [self.sp.eos_id()]

        # 截断到最大长度
        src_tokens = src_tokens[:self.max_length]
        tgt_tokens = tgt_tokens[:self.max_length]

        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


def collate_fn(batch, pad_idx: int = 0):
    """DataLoader的批处理函数"""
    src_batch, tgt_batch = zip(*batch)

    # 填充序列
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_idx
    )

    return src_padded, tgt_padded


def get_data_loader(batch_size: int, max_length: int = 256, split: str = 'train',
                    vocab_size: int = 30000, rebuild_vocab: bool = False):
    dataset = IWSLT2017Dataset(split=split, max_length=max_length,
                               vocab_size=vocab_size, rebuild_vocab=rebuild_vocab)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        collate_fn=lambda x: collate_fn(x, pad_idx=0)
    )