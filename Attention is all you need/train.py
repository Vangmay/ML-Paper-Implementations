import argparse
import torch
import torch.nn as nn

from model import Transformer
from data import build_dataloaders, make_src_mask, make_tgt_mask, PAD_IDX


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
            self.current_step * self.warmup_steps ** -1.5,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, tgt_vocab_size):
    model.train()
    losses = []
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt[:, :-1])

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, tgt_vocab_size):
    model.eval()
    losses = []
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt[:, :-1])
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--save-path", type=str, default="transformer.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bundle = build_dataloaders(
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
    )
    tokenizer_en = bundle["tokenizer_en"]
    tokenizer_de = bundle["tokenizer_de"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]

    src_vocab_size = tokenizer_en.get_vocab_size()
    tgt_vocab_size = tokenizer_de.get_vocab_size()

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, tgt_vocab_size)
        val_loss = evaluate(model, val_loader, criterion, device, tgt_vocab_size)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
        },
        args.save_path,
    )
    tokenizer_en.save_model(".", "tokenizer_en")
    tokenizer_de.save_model(".", "tokenizer_de")
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
