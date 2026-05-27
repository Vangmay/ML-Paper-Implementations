import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.epsilon = eps
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        (var, mean) = torch.var_mean(x, dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        x = (self.gamma * x) + self.beta
        return x


class PositionEmbeddings(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).reshape(max_seq_len, 1)
        denom = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch, q_len, d_model = q.shape
        k_len = k.size(1)
        v_len = v.size(1)

        Q = self.query(q).reshape(batch, q_len, self.n_heads, self.head_size).transpose(1, 2)
        K = self.key(k).reshape(batch, k_len, self.n_heads, self.head_size).transpose(1, 2)
        V = self.value(v).reshape(batch, v_len, self.n_heads, self.head_size).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_size ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = scores @ V
        scores = scores.transpose(1, 2).reshape(batch, q_len, d_model)
        return self.proj(scores)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(d_model, n_heads)
        self.FFN = FeedForward(d_model, d_ff, dropout)
        self.LN1 = LayerNorm(d_model)
        self.LN2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.LN1(x + self.multiHeadAttention(x, x, x, mask))
        x = self.LN2(x + self.FFN(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(d_model, n_heads)
        self.crossAttention = MultiHeadAttention(d_model, n_heads)
        self.FFN = FeedForward(d_model, d_ff, dropout)
        self.LN1 = LayerNorm(d_model)
        self.LN2 = LayerNorm(d_model)
        self.LN3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        x = self.LN1(x + self.multiHeadAttention(x, x, x, self_attention_mask))
        x = self.LN2(x + self.crossAttention(x, encoder_output, encoder_output, cross_attention_mask))
        x = self.LN3(x + self.FFN(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, d_model)
        self.position_embedding_layer = PositionEmbeddings(d_model, max_seq_len)
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.LN = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        embeddings = self.token_embedding_layer(x)
        embeddings = self.position_embedding_layer(embeddings)
        embeddings = self.dropout(embeddings)
        logits = embeddings
        for layer in self.encoder_stack:
            logits = layer(logits, mask)
        logits = self.LN(logits)
        return logits


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, d_model)
        self.position_embedding_layer = PositionEmbeddings(d_model, max_seq_len)
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.LN = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        embeddings = self.token_embedding_layer(x)
        embeddings = self.position_embedding_layer(embeddings)
        embeddings = self.dropout(embeddings)
        logits = embeddings
        for layer in self.decoder_stack:
            logits = layer(logits, encoder_output, tgt_mask, src_mask)
        logits = self.LN(logits)
        return logits


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.Encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len)
        self.Decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len)
        self.Proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.Encoder(src, src_mask)
        logits = self.Decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.Proj(logits)
