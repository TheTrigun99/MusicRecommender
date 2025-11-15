import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from recuperinfos import recup_playlists,trackfromplaylist
import json



def trouversp():
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
    return sp

def allsongs(fjson):
    allsong=[]
    with open(fjson, "r", encoding="utf-8") as f:
        playlists = json.load(f)
    sp=trouversp()
    for i in playlists:
        url=i["external_urls"]['spotify']
        allsong.append(trackfromplaylist(sp,url))
        print(url)
    return allsong
sp = trouversp()
print("CLIENT_ID .env =", os.getenv("SPOTIFY_CLIENT_ID"))
print("REDIRECT_URI .env =", os.getenv("SPOTIFY_REDIRECT_URI"))
print("CLIENT_ID utilisé par Spotipy =", sp.auth_manager.client_id)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def allsongss(fjson):
    with open(fjson, "r", encoding="utf-8") as f:
        playlists = json.load(f)


    futures = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for pl in playlists:
            url = pl["external_urls"]["spotify"]
            futures.append(ex.submit(trackfromplaylist, sp, url))

        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    return results

playlist=allsongss('playlists_meta.json')
with open("playlistsasons.jsonl", "w", encoding="utf-8") as f:
    for pl in playlist:
        f.write(json.dumps(pl) + "\n")
