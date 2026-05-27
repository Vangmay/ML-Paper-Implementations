import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset


PAD_IDX = 0


def write_corpus_files(train_data, en_path="en_data.txt", de_path="de_data.txt"):
    with open(en_path, "w") as f:
        for item in train_data:
            f.write(item["en"].lower() + "\n")
    with open(de_path, "w") as f:
        for item in train_data:
            f.write(item["de"].lower() + "\n")
    return en_path, de_path


def train_tokenizers(en_path="en_data.txt", de_path="de_data.txt", vocab_size=8000):
    tokenizer_en = ByteLevelBPETokenizer()
    tokenizer_en.train(
        files=[en_path],
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
    )
    tokenizer_de = ByteLevelBPETokenizer()
    tokenizer_de.train(
        files=[de_path],
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
    )
    return tokenizer_en, tokenizer_de


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
        return (
            torch.tensor(encoded_src[: self.max_seq_len]),
            torch.tensor(encoded_tgt[: self.max_seq_len]),
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def make_src_mask(src, pad_idx=PAD_IDX):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)


def make_tgt_mask(tgt, pad_idx=PAD_IDX):
    tgt_len = tgt.shape[1]
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    return pad_mask & causal_mask  # (batch, 1, seq_len, seq_len)


def build_dataloaders(max_seq_len=100, batch_size=32, vocab_size=8000):
    dataset = load_dataset("bentrevett/multi30k")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    write_corpus_files(train_data)
    tokenizer_en, tokenizer_de = train_tokenizers(vocab_size=vocab_size)

    train_dataset = TranslationDataset(train_data, tokenizer_en, tokenizer_de, max_seq_len)
    val_dataset = TranslationDataset(val_data, tokenizer_en, tokenizer_de, max_seq_len)
    test_dataset = TranslationDataset(test_data, tokenizer_en, tokenizer_de, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return {
        "tokenizer_en": tokenizer_en,
        "tokenizer_de": tokenizer_de,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
