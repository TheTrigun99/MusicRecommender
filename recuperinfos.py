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