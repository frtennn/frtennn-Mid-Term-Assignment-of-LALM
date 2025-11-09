import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .utils import positional_encoding


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        batch_size, q_seq_len = q.size(0), q.size(1)
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        # 线性变换并重塑为多头形式
        Q = self.w_q(q).view(batch_size, q_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, v_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # 确保mask的形状正确
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 从 (batch_size, seq_len, seq_len) 到 (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力到值向量
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )

        return self.w_o(output), attn_weights


class PositionWiseFFN(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class DecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None):
        # 自注意力
        self_attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 交叉注意力
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class Encoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, vocab_size: int, max_seq_length: int, d_model: int,
                 n_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_pos_encoding: bool = True):
        seq_len = x.size(1)
        # 词嵌入
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        # 位置编码（如果启用）
        if use_pos_encoding:
            x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    # def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    #     seq_len = x.size(1)
    #     # 词嵌入 + 位置编码
    #     x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
    #     x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
    #     x = self.dropout(x)
    #     # 通过编码器层
    #     for layer in self.layers:
    #         x = layer(x, mask)
    #     return self.norm(x)


class Decoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, vocab_size: int, max_seq_length: int, d_model: int,
                 n_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    # def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
    #             self_mask: Optional[torch.Tensor] = None,
    #             cross_mask: Optional[torch.Tensor] = None):
    #     seq_len = x.size(1)
    #     # 词嵌入 + 位置编码
    #     x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
    #     x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
    #     x = self.dropout(x)
    #     # 通过解码器层
    #     for layer in self.layers:
    #         x = layer(x, enc_output, self_mask, cross_mask)
    #     x = self.norm(x)
    #     return self.output_proj(x)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None,
                use_pos_encoding: bool = True):
        seq_len = x.size(1)
        # 词嵌入
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        # 位置编码（如果启用）
        if use_pos_encoding:
            x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
        x = self.norm(x)
        return self.output_proj(x)


class Transformer(nn.Module):
    """完整的Transformer序列到序列模型"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 max_seq_length: int, d_model: int = 512, n_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, max_seq_length, d_model, n_heads,
            num_encoder_layers, d_ff, dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size, max_seq_length, d_model, n_heads,
            num_decoder_layers, d_ff, dropout
        )

        # 如果源和目标词汇表相同，共享权重
        if src_vocab_size == tgt_vocab_size:
            self.decoder.token_embedding.weight = self.encoder.token_embedding.weight

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """使用Xavier均匀初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                use_pos_encoding: bool = True):
        enc_output = self.encoder(src, src_mask, use_pos_encoding)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask, use_pos_encoding)
        return dec_output

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, use_pos_encoding: bool = True):
        return self.encoder(src, src_mask, use_pos_encoding)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None,
               use_pos_encoding: bool = True):
        return self.decoder(tgt, memory, tgt_mask, memory_mask, use_pos_encoding)

    # def forward(self, src: torch.Tensor, tgt: torch.Tensor,
    #             src_mask: Optional[torch.Tensor] = None,
    #             tgt_mask: Optional[torch.Tensor] = None,
    #             memory_mask: Optional[torch.Tensor] = None):
    #     enc_output = self.encoder(src, src_mask)
    #     dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
    #     return dec_output
    #
    # def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
    #     return self.encoder(src, src_mask)
    #
    # def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
    #            tgt_mask: Optional[torch.Tensor] = None,
    #            memory_mask: Optional[torch.Tensor] = None):
    #     return self.decoder(tgt, memory, tgt_mask, memory_mask)


# 在文件末尾添加以下代码

class AveragePooling(nn.Module):
    """平均池化，用于替换注意力机制（消融实验用）"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # 对值向量进行平均池化
        batch_size, seq_len = v.size(0), v.size(1)

        # 对序列维度进行平均池化
        pooled = v.mean(dim=1, keepdim=True)  # (batch_size, 1, d_model)
        # 扩展到原始序列长度
        output = pooled.expand(batch_size, seq_len, self.d_model)

        return self.output_proj(output), None


class IdentityFFN(nn.Module):
    """恒等前馈网络，用于替换原始FFN（消融实验用）"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # 直接返回输入，不做任何变换