import argparse
import torch
from tokenizers import ByteLevelBPETokenizer

from model import Transformer
from data import make_src_mask, make_tgt_mask


@torch.no_grad()
def greedy_decode(model, src_ids, tokenizer_tgt, max_len, device):
    model.eval()
    sos_id = tokenizer_tgt.token_to_id("<sos>")
    eos_id = tokenizer_tgt.token_to_id("<eos>")

    src = torch.tensor(src_ids, device=device).unsqueeze(0)
    src_mask = make_src_mask(src)
    memory = model.Encoder(src, src_mask)

    ys = torch.tensor([[sos_id]], device=device)
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys)
        out = model.Decoder(ys, memory, src_mask, tgt_mask)
        logits = model.Proj(out[:, -1, :])
        next_token = logits.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == eos_id:
            break
    return ys[0].tolist()


def translate(model, sentence, tokenizer_src, tokenizer_tgt, device, max_len=100):
    sos_id = tokenizer_src.token_to_id("<sos>")
    eos_id = tokenizer_src.token_to_id("<eos>")
    src_ids = [sos_id] + tokenizer_src.encode(sentence.lower()).ids + [eos_id]
    out_ids = greedy_decode(model, src_ids, tokenizer_tgt, max_len, device)
    # strip sos/eos for clean output
    cleaned = [t for t in out_ids if t not in (sos_id, eos_id)]
    return tokenizer_tgt.decode(cleaned)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = Transformer(
        src_vocab_size=ckpt["src_vocab_size"],
        tgt_vocab_size=ckpt["tgt_vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_seq_len=cfg["max_seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="transformer.pt")
    parser.add_argument("--tokenizer-en-prefix", type=str, default="tokenizer_en")
    parser.add_argument("--tokenizer-de-prefix", type=str, default="tokenizer_de")
    parser.add_argument("--text", type=str, required=True, help="English sentence to translate")
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)

    tokenizer_en = ByteLevelBPETokenizer(
        f"{args.tokenizer_en_prefix}-vocab.json",
        f"{args.tokenizer_en_prefix}-merges.txt",
    )
    tokenizer_de = ByteLevelBPETokenizer(
        f"{args.tokenizer_de_prefix}-vocab.json",
        f"{args.tokenizer_de_prefix}-merges.txt",
    )

    translation = translate(model, args.text, tokenizer_en, tokenizer_de, device, max_len=args.max_len)
    print(f"EN: {args.text}")
    print(f"DE: {translation}")


if __name__ == "__main__":
    main()
