"""
Implementation of an Item2Vec model (skip-gram + negative sampling) dedicated to playlists.

The class below wraps:
- Vocabulary construction / filtering (via build.build_vocab)
- Skip-gram pair generation
- PyTorch model + training loop
- Simple helper methods to query embeddings or get nearest neighbors

Example:
    from loading import load500
    playlists = load500("spotify_dataset.csv")
    model = Item2Vec(embedding_dim=128, window_size=2, num_negatives=8)
    model.fit(playlists, epochs=5, batch_size=2048)
    print(model.most_similar("Shape of You"))
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


try:
    from .build import build_noise, build_vocab
except ImportError:  # When executed as a script from repo root
    from build import build_noise, build_vocab


Playlist = List[str]


def generate_pairs(playlists: List[Playlist], track_to_idx: Dict[str, int], window_size: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for playlist in playlists:
        encoded = [track_to_idx[track] for track in playlist if track in track_to_idx]
        for center_pos, center in enumerate(encoded):
            left = max(0, center_pos - window_size)
            right = min(len(encoded), center_pos + window_size + 1)
            for pos in range(left, right):
                if pos == center_pos:
                    continue
                pairs.append((center, encoded[pos]))
    return pairs


class SkipGramDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[int, int]],
        noise_distribution,
        num_negatives: int,
    ) -> None:
        if not pairs:
            raise ValueError("No training pairs were generated; window_size may be too small.")
        self.pairs = pairs
        self.num_negatives = num_negatives
        if isinstance(noise_distribution, torch.Tensor):
            self.noise_distribution = noise_distribution.float()
        else:
            self.noise_distribution = torch.tensor(noise_distribution, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        center, positive = self.pairs[idx]
        negatives = torch.multinomial(self.noise_distribution, self.num_negatives, replacement=True)
        return center, positive, negatives


def skipgram_collate(batch):
    centers, positives, negatives = zip(*batch)
    centers = torch.tensor(centers, dtype=torch.long)
    positives = torch.tensor(positives, dtype=torch.long)
    negatives = torch.stack(negatives).long()
    return centers, positives, negatives


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.input_embeddings.embedding_dim)
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, centers: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        center_vec = self.input_embeddings(centers)
        positive_vec = self.output_embeddings(positives)

        pos_score = torch.sum(center_vec * positive_vec, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_vec = self.output_embeddings(negatives)  # (batch, num_neg, dim)
        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self) -> torch.Tensor:
        return self.input_embeddings.weight.detach()


class Item2Vec:
    """
    Wraps a Skip-gram w/ negative sampling model for playlist recommendation use-cases.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        window_size: int = 2,
        num_negatives: int = 5,
        min_count: int = 3,
        learning_rate: float = 0.01,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed

        self.track_to_idx: Dict[str, int] | None = None
        self.idx_to_track: List[str] | None = None
        self.noise_distribution: torch.Tensor | None = None
        self.model: SkipGramNS | None = None

    def fit(
        self,
        playlists: Sequence[Playlist],
        epochs: int = 5,
        batch_size: int = 2048,
        shuffle_pairs: bool = True,
        verbose: bool = True,
    ) -> None:
        dataset = self._prepare_training_data(playlists, shuffle_pairs=shuffle_pairs)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=skipgram_collate,
            num_workers=0,
        )

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.model = SkipGramNS(len(self.track_to_idx), self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_examples = 0

            for centers, positives, negatives in loader:
                centers = centers.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)

                optimizer.zero_grad()
                loss = self.model(centers, positives, negatives)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * centers.size(0)
                total_examples += centers.size(0)

            if verbose:
                avg_loss = total_loss / max(1, total_examples)
                print(f"[Epoch {epoch}/{epochs}] loss={avg_loss:.4f}")

    def _prepare_training_data(self, playlists: Sequence[Playlist], shuffle_pairs: bool) -> SkipGramDataset:
        if not playlists:
            raise ValueError("No playlists provided to Item2Vec.fit.")

        track_to_idx, idx_to_track, counts, filtered_playlists = build_vocab(playlists, self.min_count)
        if not track_to_idx:
            raise ValueError("Vocabulary is empty after min_count filtering.")

        self.track_to_idx = track_to_idx
        self.idx_to_track = idx_to_track
        noise = build_noise(counts)
        noise_dist = torch.tensor(noise, dtype=torch.float32)
        self.noise_distribution = noise_dist

        pairs = generate_pairs(filtered_playlists, track_to_idx, self.window_size)
        if shuffle_pairs:
            random.shuffle(pairs)

        return SkipGramDataset(pairs, noise_dist, self.num_negatives)

    def get_vector(self, track: str) -> torch.Tensor:
        if self.model is None or self.track_to_idx is None:
            raise RuntimeError("The model must be trained before requesting embeddings.")
        if track not in self.track_to_idx:
            raise KeyError(f"'{track}' is not part of the vocabulary.")
        idx = self.track_to_idx[track]
        embeddings = self.model.get_embeddings().to(self.device)
        return embeddings[idx]

    def most_similar(self, track: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.model is None or self.track_to_idx is None or self.idx_to_track is None:
            raise RuntimeError("Train the model before calling most_similar.")
        if track not in self.track_to_idx:
            raise KeyError(f"'{track}' is not part of the vocabulary.")

        embeddings = self.model.get_embeddings()
        embeddings = F.normalize(embeddings, dim=1)
        idx = self.track_to_idx[track]
        query = embeddings[idx]
        scores = torch.matmul(embeddings, query)
        top_indices = torch.topk(scores, top_k + 1).indices.tolist()

        neighbors: List[Tuple[str, float]] = []
        for cand_idx in top_indices:
            candidate = self.idx_to_track[cand_idx]
            if candidate == track:
                continue
            neighbors.append((candidate, float(scores[cand_idx].item())))
            if len(neighbors) == top_k:
                break
        return neighbors

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        artifact = {
            "state_dict": self.model.state_dict(),
            "track_to_idx": self.track_to_idx,
            "idx_to_track": self.idx_to_track,
            "config": {
                "embedding_dim": self.embedding_dim,
                "window_size": self.window_size,
                "num_negatives": self.num_negatives,
                "min_count": self.min_count,
                "learning_rate": self.learning_rate,
            },
        }
        torch.save(artifact, path)
        print(f"Item2Vec saved to {Path(path).resolve()}")

    def load(self, path: str | Path, map_location: str | None = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        config = checkpoint.get("config", {})
        self.embedding_dim = config.get("embedding_dim", self.embedding_dim)
        self.window_size = config.get("window_size", self.window_size)
        self.num_negatives = config.get("num_negatives", self.num_negatives)
        self.min_count = config.get("min_count", self.min_count)
        self.learning_rate = config.get("learning_rate", self.learning_rate)

        self.track_to_idx = checkpoint["track_to_idx"]
        self.idx_to_track = checkpoint["idx_to_track"]

        self.model = SkipGramNS(len(self.track_to_idx), self.embedding_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        print(f"Item2Vec loaded from {Path(path).resolve()}")


if __name__ == "__main__":
    try:
        from loading import load500
    except ImportError:
        from .loading import load500  # type: ignore

    playlists = load500("spotify_dataset.csv")
    print(f"Playlists loaded: {len(playlists)}")
    item2vec = Item2Vec(embedding_dim=64, window_size=2, num_negatives=5)
    item2vec.fit(playlists, epochs=1, batch_size=1024)
    sample_track = item2vec.idx_to_track[0] if item2vec.idx_to_track else None
    if sample_track:
        print(f"Neighbors for '{sample_track}':")
        for neighbor, score in item2vec.most_similar(sample_track, top_k=5):
            print(f"  {neighbor} ({score:.3f})")
