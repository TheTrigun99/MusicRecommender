import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from recuperinfos import recup_playlists,trackfromplaylist

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

print(trackfromplaylist(sp,"https://open.spotify.com/playlist/7E3uEa1emOcbZJuB8sFXeK"))

