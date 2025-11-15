from requests.exceptions import HTTPError
import time
def recupliked(sp_client):
    """Pagine sur toute la librairie 'Liked Songs' (50 par page)."""
    titres = []
    limit = 50 #spot limite à 50 sons par requête !
    count = 0 
    while True:
        try:
            page = sp_client.current_user_saved_tracks(limit=limit, offset=count)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 429: #erreur 429 consiste à l'érreur quand on fait trop de reqîete à la fois (rate)
                retry_after = int(e.response.headers.get("Retry-After", "2")) #il faut pateinter le temps demandé
                time.sleep(retry_after + 1)
                continue
            raise
        batch = page.get("items", [])
        if not batch:
            break
        titres.extend(batch)
        count += limit #on récupère les sons 0-49 puis 50-99 etc
    return titres

def audiofeatures(sp,titres):
    '''
    Récupères les audio features, malheuresement obsolète car spotify bloque ceci depuis 2024'''
    idss=[titre['track']['id'] for titre in titres]
    audio_features=[]
    for i in range(0,len(idss),100): # 100 sons par reqûete (limite)
        while True:
            try: 
                page = sp.audio_features(idss[i:i+100])
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 429: #erreur 429 consiste à l'érreur quand on fait trop de reqîete à la fois (rate)
                    retry_after = int(e.response.headers.get("Retry-After", "2")) #il faut pateinter le temps demandé
                    time.sleep(retry_after + 1)
                    continue
                raise
            if not batch:
                break
            audio_features=audio_features.extend(batch)

    return audio_features 


def spot_call(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)

        except HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", "2"))
                print(f"[429] Rate limit → attente obligatoire : {retry_after} sec")
                time.sleep(retry_after + 1)
                continue
            raise

        except Exception as e:
            print("Erreur réseau temporaire, retry dans 0.2s:", e)
            time.sleep(0.2)
            continue

def recup_playlists(sp,queries,max_playlists=1000):
    '''
    récupères playlists spotify
    '''
    vus = set()
    all_playlists = []

    for q in queries:
        print(f"\n Query: {q}")

        for offset in range(0, 1000, 50):   #on va de page en page sur spotify
            if len(vus) >= max_playlists:   #on peut pas trop doser le nombre de playlists que spotify va trouver
                print("Limite atteinte.")   #du coup je met une limite
                return all_playlists

            res = spot_call(sp.search, q=q, type="playlist", limit=50, offset=offset)
            items = res["playlists"]["items"]

            if not items:
                break

            for p in items:
                if p is None:
                    continue
                pid = p["id"]
                if pid not in vus:
                    vus.add(pid)
                    all_playlists.append(p)

            # micro pause pour réduire les risques de 429
            time.sleep(0.2)
        print(len(all_playlists))
    return all_playlists 

def traiter(res,t):
    for item in res['items']:
        track = item.get('track')
        if track and track.get('id'):
            t.append(track['id'])
    return
def trackfromplaylist(sp, p):
    tracks = []
    res = spot_call(sp.playlist_items, p, limit=100, offset=0)
    traiter(res, tracks)
    while res['next']:
        res = spot_call(sp.next, res)
        traiter(res, tracks)

    return tracks
            
