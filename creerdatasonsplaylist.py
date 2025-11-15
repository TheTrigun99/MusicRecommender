import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from recuperinfos import recup_playlists,trackfromplaylist
import json

import time

from dotenv import load_dotenv


def trouversp():
    load_dotenv()  # charge .env proprement
    
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope="playlist-read-private playlist-read-collaborative"
    ))

'''def allsongs(fjson):
    allsong=[]
    with open(fjson, "r", encoding="utf-8") as f:
        playlists = json.load(f)
    sp=trouversp()
    for i in playlists:
        url=i["external_urls"]['spotify']
        allsong.append(trackfromplaylist(sp,url))
        print(url)
    return allsong'''

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def allsongss_final(fjson):
    sp = trouversp()
    
    with open(fjson, "r", encoding="utf-8") as f:
        playlists = json.load(f)
    playlists = playlists[:1000]
    results = []
    for pl in tqdm(playlists):
        url = pl["external_urls"]["spotify"]
        try:
            res = trackfromplaylist(sp, url)
            results.append(res)
        except Exception as e:
            print("Erreur:", e)
            time.sleep(2)
            sp = trouversp()   # recreer client propre en cas d’erreur
            continue

        time.sleep(1)    # rate limit pour éviter ban

    return results


'''def allsongss(fjson):
    with open(fjson, "r", encoding="utf-8") as f:
        playlists = json.load(f)


    futures = []
    with ThreadPoolExecutor(max_workers=1) as ex:
        futures = []
        for pl in playlists:
            url = pl["external_urls"]["spotify"]
            futures.append(ex.submit(trackfromplaylist, sp, url))
            time.sleep(0.1)   

        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    return results'''

playlist=allsongss_final('playlists_meta.json')
with open("playlistsasons.jsonl", "w", encoding="utf-8") as f:
    for pl in playlist:
        f.write(json.dumps(pl) + "\n")
