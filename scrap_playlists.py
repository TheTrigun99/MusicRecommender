"""
Scrape Spotify playlists that actually contain the locally embedded tracks.

Usage examples:
    python scrap_playlists.py --token <your-oauth-token>
    python scrap_playlists.py --max-tracks 100 --per-track 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials


def load_env_file(env_path: str = ".env") -> None:
    """Populate os.environ with variables declared in a .env file."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def build_spotify_client(token: Optional[str]) -> spotipy.Spotify:
    """Return a Spotify client that prefers a user-provided token."""
    if token:
        return spotipy.Spotify(auth=token)

    load_env_file()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Spotify credentials. Provide --token or define "
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in the .env file."
        )
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth_manager)


def spotify_call(func, *args, max_retries: int = 5, **kwargs):
    """Wrapper that retries Spotify calls on rate limits or transient errors."""
    attempt = 0
    backoff = 1.0
    while True:
        try:
            return func(*args, **kwargs)
        except SpotifyException as exc:  # pragma: no cover - thin network wrapper
            attempt += 1
            if exc.http_status == 429:
                retry_after = int(exc.headers.get("Retry-After", "1"))
                time.sleep(retry_after + 0.5)
            elif exc.http_status in {500, 502, 503, 504} and attempt <= max_retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                raise
        if attempt > max_retries:
            raise RuntimeError(f"Spotify call failed after {max_retries} retries: {func.__name__}")


def safe_term(term: str) -> str:
    """Return a Spotify search-friendly quoted term."""
    escaped = term.replace('"', "\\\"")
    return f'"{escaped}"'


def load_embeddings_meta(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load the embeddings metadata file and optionally trim the track list."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected format in {path}, expected a list of tracks.")
    if limit is not None:
        return data[:limit]
    return data


def guess_local_title(entry: Dict[str, Any]) -> Optional[str]:
    """Return the best-guess track title from metadata."""
    title = entry.get("title")
    if title:
        return title.strip()
    path = entry.get("path")
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    return None


def search_track(sp: spotipy.Spotify, title: str) -> Optional[Dict[str, Any]]:
    """Search Spotify for a single track that matches the provided title."""
    query = f"track:{safe_term(title)}"
    response = spotify_call(sp.search, q=query, type="track", limit=1)
    tracks = (response.get("tracks") or {}).get("items") or []
    return tracks[0] if tracks else None


def search_candidate_playlists(
    sp: spotipy.Spotify, track_name: str, artist_name: Optional[str], limit: int
) -> List[Dict[str, Any]]:
    """Search playlist candidates based on track name and artist."""
    query = safe_term(track_name)
    if artist_name:
        query = f"{query} {safe_term(artist_name)}"
    response = spotify_call(sp.search, q=query, type="playlist", limit=limit)
    return (response.get("playlists") or {}).get("items") or []


def playlist_contains_track(
    sp: spotipy.Spotify,
    playlist_id: str,
    track_id: str,
    max_items: int,
) -> bool:
    """Check whether a playlist really contains the target track."""
    fetched = 0
    offset = 0
    while fetched < max_items:
        batch_limit = min(100, max_items - fetched)
        try:
            page = spotify_call(
                sp.playlist_items,
                playlist_id,
                offset=offset,
                limit=batch_limit,
                fields="items.track.id,items.track.name,items.track.artists.name,total,next",
            )
        except SpotifyException as exc:
            if exc.http_status in {401, 403, 404}:
                return False
            raise
        items = page.get("items") or []
        if not items:
            break
        for item in items:
            track = item.get("track") or {}
            if track.get("id") == track_id:
                return True
        fetched += len(items)
        offset += len(items)
        if not page.get("next"):
            break
    return False


def scrape_playlists(
    sp: spotipy.Spotify,
    embeddings: Iterable[Dict[str, Any]],
    per_track_limit: int,
    playlist_search_limit: int,
    max_playlist_scan: int,
) -> List[Dict[str, Any]]:
    """Collect playlists that contain tracks from the embeddings metadata."""
    aggregated: Dict[str, Dict[str, Any]] = {}
    embeddings_list = list(embeddings)
    for index, entry in enumerate(embeddings_list, start=1):
        local_title = guess_local_title(entry)
        if not local_title:
            continue
        spotify_track = search_track(sp, local_title)
        if not spotify_track:
            print(f"[{index}/{len(embeddings_list)}] Aucun resultat sur Spotify pour '{local_title}'.")
            continue
        track_id = spotify_track["id"]
        track_name = spotify_track["name"]
        artist_name = spotify_track["artists"][0]["name"] if spotify_track.get("artists") else ""
        candidates = search_candidate_playlists(sp, track_name, artist_name, playlist_search_limit)
        matched_for_track = 0
        for playlist in candidates:
            if playlist is None:
                continue
            playlist_id = playlist.get("id")
            if not playlist_id:
                continue
            if not playlist_contains_track(sp, playlist_id, track_id, max_playlist_scan):
                continue

            playlist_entry = aggregated.setdefault(
                playlist_id,
                {
                    "id": playlist_id,
                    "name": playlist.get("name"),
                    "description": playlist.get("description"),
                    "owner": (playlist.get("owner") or {}).get("display_name")
                    or (playlist.get("owner") or {}).get("id"),
                    "external_url": (playlist.get("external_urls") or {}).get("spotify"),
                    "image": (playlist.get("images") or [{}])[0].get("url") if playlist.get("images") else None,
                    "total_tracks": (playlist.get("tracks") or {}).get("total"),
                    "matched_tracks": [],
                },
            )
            if any(match["local_title"] == local_title for match in playlist_entry["matched_tracks"]):
                continue
            playlist_entry["matched_tracks"].append(
                {
                    "local_title": local_title,
                    "spotify_track": track_name,
                    "spotify_artists": [artist["name"] for artist in spotify_track.get("artists", [])],
                }
            )
            matched_for_track += 1
            if matched_for_track >= per_track_limit:
                break
        print(
            f"[{index}/{len(embeddings_list)}] '{local_title}' -> {matched_for_track} playlist(s) validees."
        )

    for playlist in aggregated.values():
        playlist["matched_track_count"] = len(playlist["matched_tracks"])
    ordered = sorted(aggregated.values(), key=lambda item: item["matched_track_count"], reverse=True)
    return ordered


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Spotify playlists connected to your embedded local tracks."
    )
    parser.add_argument(
        "--metadata",
        default=os.path.join("embeddings", "embeddings_meta.json"),
        help="Chemin vers le fichier metadata genere par embedextract.py.",
    )
    parser.add_argument(
        "--output",
        default="playlists_from_embeddings.json",
        help="Fichier ou les playlists retenues seront sauvegardees.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limiter le nombre de morceaux pris en compte (utile pour des essais rapides).",
    )
    parser.add_argument(
        "--per-track",
        type=int,
        default=10,
        help="Nombre maximum de playlists a conserver par morceau local.",
    )
    parser.add_argument(
        "--playlist-search-limit",
        type=int,
        default=10,
        help="Nombre de playlists candidates a examiner pour chaque morceau.",
    )
    parser.add_argument(
        "--max-playlist-scan",
        type=int,
        default=250,
        help="Nombre maximum de pistes a parcourir dans chaque playlist lors de la verification.",
    )
    parser.add_argument(
        "--token",
        help="Jeton OAuth Spotify a utiliser directement si vous en avez deja un.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    embeddings = load_embeddings_meta(args.metadata, args.max_tracks)
    sp_client = build_spotify_client(args.token)
    playlists = scrape_playlists(
        sp_client,
        embeddings,
        per_track_limit=args.per_track,
        playlist_search_limit=args.playlist_search_limit,
        max_playlist_scan=args.max_playlist_scan,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(playlists, handle, ensure_ascii=False, indent=2)
    print(f"{len(playlists)} playlist(s) sauvegardees dans {args.output}.")


if __name__ == "__main__":
    main(sys.argv[1:])
