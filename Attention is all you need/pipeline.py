from datasets import load_dataset
import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
import math

"""# Core architecture"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
dropout = 0.1
max_seq_len = 5000

class LayerNorm(nn.Module):
  def __init__(self, d_model, eps=1e-5):
    super().__init__()
    self.epsilon = eps
    self.beta = nn.Parameter(torch.zeros(d_model))
    self.gamma = nn.Parameter(torch.ones(d_model))

  def forward(self, x):
    (var, mean) = torch.var_mean(x, dim=-1, keepdim=True)
    x = (x-mean) / torch.sqrt(var + self.epsilon)
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
    self.register_buffer('pe', pe)

  def forward(self, x):
    return x + self.pe[:x.size(1)]

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.n_heads = n_heads
    self.head_size = d_model // n_heads

    self.key = nn.Linear(d_model, d_model, bias=False)
    self.query = nn.Linear(d_model, d_model, bias=False)
    self.value = nn.Linear(d_model, d_model, bias=False)
    self.proj = nn.Linear(d_model, d_model)

  def forward(self, q, k, v, mask = None):
    (batch, seq_len, d_model) = q.shape

    Q = self.query(q).reshape(batch, self.n_heads, seq_len, self.head_size).transpose(1, 2)
    K = self.key(k).reshape(batch, self.n_heads, seq_len, self.head_size).transpose(1, 2)
    V = self.value(v).reshape(batch, self.n_heads, seq_len, self.head_size).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / (self.head_size**0.5)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    scores = scores @ V
    scores = scores.transpose(1, 2).reshape(batch, seq_len, d_model)
    return self.proj(scores)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout):
    super().__init__()
    self.multiHeadAttention = MultiHeadAttention(d_model, n_heads)
    self.FFN = FeedForward(d_model, dropout)
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
    self.FFN = FeedForward(d_model, dropout)
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
    self.encoder_stack = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
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
    self.decoder_stack = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    self.LN = LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
    embeddings = self.token_embedding_layer(x)
    embeddings = self.position_embedding_layer(embeddings)
    embeddings = self.dropout(embeddings)
    logits = embeddings
    for layer in self.decoder_stack:
      logits = layer(logits, encoder_output, src_mask, tgt_mask)
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

"""# Data Pipeline"""

from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

dataset = load_dataset("bentrevett/multi30k")

train_data = dataset["train"]
val_data   = dataset["validation"]
test_data  = dataset["test"]

# check what you're working with
print(train_data[0])
print(len(train_data))

with open("en_data.txt", "w") as f:
    for item in train_data:
        f.write(item["en"].lower() + "\n")

with open("de_data.txt", "w") as f:
    for item in train_data:
        f.write(item["de"].lower() + "\n")

tokenizer_en = ByteLevelBPETokenizer()
tokenizer_en.train(files=["en_data.txt"], vocab_size=8000, special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"])

tokenizer_de = ByteLevelBPETokenizer()
tokenizer_de.train(files=["de_data.txt"], vocab_size=8000, special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"])

encoded = tokenizer_en.encode("Two young men are outside near bushes.")
print(encoded.tokens)
print(encoded.ids)

decoded = tokenizer_en.decode(encoded.ids)
print(decoded)

print(tokenizer_en.token_to_id("<pad>"))
print(tokenizer_en.token_to_id("<sos>"))
print(tokenizer_en.token_to_id("<eos>"))
print(tokenizer_en.token_to_id("<unk>"))

class TranslationDataset(Dataset):
  def __init__(self, data, tokenizer_en, tokenizer_de, max_seq_len):
    super().__init__()
    self.data = data
    self.tokenizer_en = tokenizer_en
    self.tokenizer_de = tokenizer_de
    self.max_seq_len = max_seq_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    src, tgt = item["en"], item["de"]
    sos_id = self.tokenizer_en.token_to_id("<sos>")
    eos_id = self.tokenizer_en.token_to_id("<eos>")
    encoded_src = [sos_id] + self.tokenizer_en.encode(src).ids + [eos_id]
    encoded_tgt = [sos_id] + self.tokenizer_de.encode(tgt).ids + [eos_id]
    return torch.tensor(encoded_src[:self.max_seq_len]), torch.tensor(encoded_tgt[:self.max_seq_len])

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

train_dataset = TranslationDataset(train_data, tokenizer_en, tokenizer_de, max_seq_len=100)
val_dataset   = TranslationDataset(val_data, tokenizer_en, tokenizer_de, max_seq_len=100)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

src, tgt = next(iter(train_loader))
print(src.shape, tgt.shape)

def make_src_mask(src, pad_idx=0):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

def make_tgt_mask(tgt, pad_idx=0):
    tgt_len = tgt.shape[1]
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()  # (seq_len, seq_len)
    return pad_mask & causal_mask  # (batch, 1, seq_len, seq_len)

src_vocab_size=tokenizer_en.get_vocab_size()
    tgt_vocab_size=tokenizer_de.get_vocab_size()

model = Transformer(
    src_vocab_size=tokenizer_en.get_vocab_size(),
    tgt_vocab_size=tokenizer_de.get_vocab_size(),
    d_model=512, n_layers=6, n_heads=8,
    d_ff=2048, dropout=0.1, max_seq_len=100
).to(device)

"""# Training Loop"""

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = (self.d_model ** -0.5) * min(
            self.current_step ** -0.5,
            self.current_step * self.warmup_steps ** -1.5
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is <pad>
loss_list = []

max_epochs = 10000

for i in range(max_epochs):
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input, src_mask, tgt_mask)

        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
        loss_list.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

