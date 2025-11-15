import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from recuperinfos import recup_playlists
scope = "user-library-read"
with open(".env") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=", 1)
            os.environ[key] = value

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
    scope=scope
))

'''results = sp.current_user_saved_tracks()
for idx, item in enumerate(results['items']):
    track = item['track']
    print(idx, track['artists'][0]['name'], " – ", track['name'])'''
#on récup tous mes titres likés (on va les utiliser pour pas trop avoir des sons biaisés et que
#le travail de recommandation soit déjà fait. J'ai écouté plusieurs genres durant mes 4 ans sur spotify...
#titres = recupliked(sp)
#print(f" On a {len(titres)} titres récupérés.")
queries=[    "top hits",
    "popular songs",
    "best of",
    "all time hits",
    "french hits",
    "rock hits","classic hits","party","summer hits","Rock Classics","Top 50 Global","love songs hits","dance hits","classical hits"]
playlists=recup_playlists(sp,queries,max_playlists=8000)
import json
print(len(playlists))
with open("playlists_meta.json", "w", encoding="utf-8") as f:
    json.dump(playlists, f, ensure_ascii=False, indent=2)