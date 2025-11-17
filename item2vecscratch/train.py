"""
Training script for the minimalist Item2VecPaper model (skip-gram + negative sampling).

Example:
    python -m item2vecscratch.train --data spotify_dataset.csv --epochs 5 --output item2vec.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .build import build_noise, build_vocab, gen_couples
from .loading import load500
from .modele import Item2VecPaper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Item2VecPaper embeddings on playlist data.")
    parser.add_argument("--data", type=Path, default=Path("spotify_dataset.csv"), help="CSV export containing playlists.")
    parser.add_argument("--output", type=Path, default=Path("item2vec.pt"), help="Where to store embeddings + vocab.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size", help="Training batch size.")
    parser.add_argument("--embedding-dim", type=int, default=128, dest="embedding_dim", help="Embedding size.")
    parser.add_argument("--window-size", type=int, default=2, dest="window_size", help="Skip-gram window on each side.")
    parser.add_argument("--negatives", type=int, default=5, help="Negative samples per positive pair.")
    parser.add_argument("--min-count", type=int, default=3, dest="min_count", help="Minimum track frequency to keep.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for Adam.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    return parser.parse_args()


def iterate_batches(pairs, batch_size):
    for idx in range(0, len(pairs), batch_size):
        yield pairs[idx : idx + batch_size]


def main() -> None:
    args = parse_args()
    playlists = load500(str(args.data))
    print(f"Playlists chargees: {len(playlists)}")
    track2id, id2track, counts, filtered = build_vocab(playlists, min_count=args.min_count)
    pairs = gen_couples(filtered, track2id, args.window_size)
    
    noise_dist = torch.tensor(build_noise(counts), dtype=torch.float32)
    noise_dist = noise_dist.to(args.device)

    model = Item2VecPaper(num_items=len(track2id), emb_dim=args.embedding_dim).to(args.device)
    model.train() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        total_examples = 0
        
        for batch in iterate_batches(pairs, args.batch_size):
            centers = torch.tensor([c for c, _ in batch], dtype=torch.long, device=args.device)
            positives = torch.tensor([p for _, p in batch], dtype=torch.long, device=args.device)
            negatives = torch.multinomial(noise_dist, args.negatives * len(batch), replacement=True)
            negatives = negatives.view(len(batch), args.negatives)

            optimizer.zero_grad()
            loss = model(centers, positives, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_examples += len(batch)

        avg_loss = total_loss / max(1, total_examples)
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg_loss:.4f}")

    artifact = {
        "embeddings": model.get_embeddings().cpu(),
        "track_to_id": track2id,
        "id_to_track": id2track,
        "config": vars(args),
    }
    torch.save(artifact, args.output)
    print(f"Model saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
