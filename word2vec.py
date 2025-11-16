"""
Training script for a Skip-Gram Word2Vec model with negative sampling on playlist data.

The file expects playlists serialized as either:
1. JSON Lines (``.jsonl``) where every line is a list of Spotify track IDs.
2. JSON files containing a list of playlists. Each playlist can be a list of track IDs
   or a dict that exposes either ``track_ids`` or a Spotify-like ``tracks`` payload.

Example usage:
    python word2vec.py --playlists playlistsasons.jsonl --epochs 10 --batch-size 2048
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


Playlist = List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec (skip-gram + negative sampling) model on playlists."
    )
    parser.add_argument(
        "--playlists",
        type=Path,
        default=Path("playlistsasons.jsonl"),
        help="Path to the playlists file (.jsonl or .json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playlist_word2vec.pt"),
        help="Destination file for the trained embeddings + metadata.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        dest="embedding_dim",
        help="Embedding size for items.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        dest="window_size",
        help="Number of neighbors to look at on each side of the center track.",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=5,
        dest="num_negatives",
        help="How many negative samples to draw per positive pair.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        dest="min_count",
        help="Minimum number of occurrences for a track to be kept in the vocab.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        dest="batch_size",
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        dest="learning_rate",
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max-playlists",
        type=int,
        default=None,
        dest="max_playlists",
        help="Optional limit on the number of playlists to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_playlists(path: Path, max_playlists: int | None = None) -> List[Playlist]:
    if not path.exists():
        raise FileNotFoundError(f"Impossible de trouver le fichier playlists: {path}")

    playlists: List[Playlist] = []

    def append_playlist(raw: Sequence[str]) -> None:
        if not raw:
            return
        playlist = [track for track in raw if isinstance(track, str) and track]
        if playlist:
            playlists.append(playlist)

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                tracks = extract_track_ids(payload)
                if tracks:
                    append_playlist(tracks)
                if max_playlists and len(playlists) >= max_playlists:
                    break
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        candidates: Iterable = payload
        if isinstance(payload, dict):
            for key in ("playlists", "items", "data"):
                if key in payload and isinstance(payload[key], list):
                    candidates = payload[key]
                    break
            else:
                candidates = [payload]

        for entry in candidates:
            tracks = extract_track_ids(entry)
            if tracks:
                append_playlist(tracks)
            if max_playlists and len(playlists) >= max_playlists:
                break

    if not playlists:
        raise RuntimeError(
            "Aucune playlist valide trouvee. Verifie le format du fichier d'entree."
        )
    return playlists


def extract_track_ids(entry) -> Playlist:
    if isinstance(entry, list):
        return [track for track in entry if isinstance(track, str) and track]

    if not isinstance(entry, dict):
        return []

    if "track_ids" in entry and isinstance(entry["track_ids"], list):
        return [tid for tid in entry["track_ids"] if isinstance(tid, str) and tid]

    if "tracks" in entry:
        tracks = entry["tracks"]
        if isinstance(tracks, dict):
            tracks = tracks.get("items") or tracks.get("tracks")
        if isinstance(tracks, list):
            extracted: Playlist = []
            for track in tracks:
                if isinstance(track, str) and track:
                    extracted.append(track)
                    continue
                if isinstance(track, dict):
                    if "track" in track and isinstance(track["track"], dict):
                        tid = track["track"].get("id")
                    else:
                        tid = track.get("id")
                    if tid:
                        extracted.append(tid)
            return extracted
    return []


def build_vocab(playlists: List[Playlist], min_count: int) -> tuple[Dict[str, int], List[str], torch.Tensor]:
    counter = Counter()
    for playlist in playlists:
        counter.update(playlist)

    tokens = [track for track, freq in counter.items() if freq >= min_count]
    if not tokens:
        raise RuntimeError(
            f"Aucun titre n'a une frequence >= {min_count}. Reduis --min-count."
        )

    tokens.sort()
    track_to_idx = {track: idx for idx, track in enumerate(tokens)}
    idx_to_track = tokens

    freqs = torch.tensor([counter[track] ** 0.75 for track in idx_to_track], dtype=torch.float)
    noise_dist = freqs / freqs.sum()
    return track_to_idx, idx_to_track, noise_dist


class SkipGramNegDataset(Dataset):
    def __init__(
        self,
        playlists: List[Playlist],
        track_to_idx: Dict[str, int],
        window_size: int,
        num_negatives: int,
        noise_dist: torch.Tensor,
    ) -> None:
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.noise_dist = noise_dist
        self.pairs: List[tuple[int, int]] = []

        for playlist in playlists:
            encoded = [track_to_idx[track] for track in playlist if track in track_to_idx]
            for center_pos, center in enumerate(encoded):
                left = max(0, center_pos - window_size)
                right = min(len(encoded), center_pos + window_size + 1)
                for context_pos in range(left, right):
                    if context_pos == center_pos:
                        continue
                    self.pairs.append((center, encoded[context_pos]))

        if not self.pairs:
            raise RuntimeError("La fenetre choisie ne produit aucune paire d'entrainement.")
        random.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        center, positive = self.pairs[idx]
        negatives = torch.multinomial(self.noise_dist, self.num_negatives, replacement=True)
        return center, positive, negatives


def skipgram_collate(batch):
    centers, positives, negatives = zip(*batch)
    centers_tensor = torch.tensor(centers, dtype=torch.long)
    positives_tensor = torch.tensor(positives, dtype=torch.long)
    negatives_tensor = torch.stack(negatives).long()
    return centers_tensor, positives_tensor, negatives_tensor


class Word2VecNS(nn.Module):
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
        pos_vec = self.output_embeddings(positives)

        pos_score = torch.sum(center_vec * pos_vec, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_vec = self.output_embeddings(negatives)  # (batch, num_neg, dim)
        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_item_embeddings(self) -> torch.Tensor:
        return self.input_embeddings.weight.detach()


def train(model: Word2VecNS, loader: DataLoader, epochs: int, device: torch.device, learning_rate: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_examples = 0
        for centers, positives, negatives in loader:
            centers = centers.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(centers, positives, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * centers.size(0)
            total_examples += centers.size(0)

        avg_loss = total_loss / max(1, total_examples)
        print(f"[Epoch {epoch}/{epochs}] loss={avg_loss:.4f}")


def save_artifacts(
    path: Path,
    model: Word2VecNS,
    idx_to_track: List[str],
    track_to_idx: Dict[str, int],
    args: argparse.Namespace,
) -> None:
    embeddings = model.get_item_embeddings().cpu()
    artifact = {
        "embeddings": embeddings,
        "idx_to_track": idx_to_track,
        "track_to_idx": track_to_idx,
        "config": vars(args),
    }
    torch.save(artifact, path)
    print(f"Embeddings sauvegardees dans {path.resolve()}")


def recommend(track_id: str, k: int, embeddings: torch.Tensor, track_to_idx: Dict[str, int], idx_to_track: List[str]) -> List[str]:
    if track_id not in track_to_idx:
        raise KeyError(f"'{track_id}' n'est pas present dans le vocabulaire.")
    vectors = F.normalize(embeddings, dim=1)
    query = vectors[track_to_idx[track_id]].unsqueeze(0)
    scores = torch.matmul(query, vectors.T).squeeze(0)
    topk = torch.topk(scores, k + 1).indices.tolist()
    recs = []
    for idx in topk:
        candidate = idx_to_track[idx]
        if candidate == track_id:
            continue
        recs.append(candidate)
        if len(recs) == k:
            break
    return recs


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    playlists = load_playlists(args.playlists, args.max_playlists)
    print(f"{len(playlists)} playlists chargees.")

    track_to_idx, idx_to_track, noise_dist = build_vocab(playlists, args.min_count)
    print(f"Vocabulaire retenu: {len(track_to_idx)} titres.")

    dataset = SkipGramNegDataset(
        playlists=playlists,
        track_to_idx=track_to_idx,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        noise_dist=noise_dist,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=skipgram_collate,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = Word2VecNS(vocab_size=len(track_to_idx), embedding_dim=args.embedding_dim)
    train(model, loader, args.epochs, device, args.learning_rate)
    save_artifacts(args.output, model, idx_to_track, track_to_idx, args)


if __name__ == "__main__":
    main()
