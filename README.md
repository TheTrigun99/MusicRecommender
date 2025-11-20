# MusicRecommender

Pipeline de recommandation musicale construit autour des API Spotify et d’un entraînement *Item2Vec* minimaliste. Le projet collecte des playlists publiques, récupère l’ensemble des morceaux associés, puis apprend des embeddings qui servent à trouver des titres similaires.

## Fonctionnalités clés
- **Collecte Spotify** : recherche de milliers de playlists publiques via Spotipy et gestion du *rate limiting* (`extraction.py`, `recuperinfos.py`).
- **Extraction des pistes** : récupération de tous les morceaux de chaque playlist et export au format JSON Lines (`creerdatasonsplaylist.py`).
- **Training Item2Vec** : implémentation from scratch (PyTorch) du modèle skip-gram avec échantillonnage négatif (`item2vecscratch/train.py`, `modele.py`).
- **Recherche de voisins** : scripts pour interroger les embeddings générés (`item2vecscratch/test.py`, `test2.py`) et explorations FAISS (`usingfaiss.py`).
- **Analyses audio** : exemples d’extraction de features via Librosa (`audio_analysys.py`).

## Organisation du dépôt
- `extraction.py` : recherche de playlists par mots-clés et sauvegarde dans `playlists_meta.json`.
- `creerdatasonsplaylist.py` : lit `playlists_meta.json`, récupère les tracks de chaque playlist et écrit `playlistsasons.jsonl`.
- `recuperinfos.py` : utilitaires Spotipy (pagination, gestion des erreurs 429, extraction de morceaux).
- `audio_analysys.py` : exemple de calcul de spectrogrammes et features audio.
- `usingfaiss.py` : démonstration de recherche par similarité avec FAISS.
- `item2vecscratch/`
  - `loading.py` : chargement CSV (`playlistname`, `trackname`) → liste de playlists.
  - `build.py` : construction du vocabulaire, couples (centre/contexte) et distribution de bruit.
  - `modele.py` : définition du modèle `Item2VecPaper`.
  - `train.py` : script CLI d’entraînement (torch).
  - `test.py` / `test2.py` : recherche de voisins avec les embeddings maison ou Gensim.
  - `treatdata.py` : placeholder pour futurs traitements de données.

## Pré-requis
- Python 3.10+.
- Compte Spotify Developer avec une application créée.
- Bibliothèques principales : `spotipy`, `python-dotenv`, `pandas`, `numpy`, `torch`, `tqdm`, `gensim`, `librosa`, `matplotlib`, `faiss-cpu`, `requests`.

Pour le .env
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=...



