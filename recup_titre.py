from requests.exceptions import HTTPError
import time
def recupliked(sp_client):
    """Pagine sur toute la librairie 'Liked Songs' (50 par page)."""
    titres = []
    limit = 50 #spot limite à 50 sons
    count = 0 
    while True:
        try:
            page = sp_client.current_user_saved_tracks(limit=limit, offset=count)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", "2"))
                time.sleep(retry_after + 1)
                continue
            raise
        batch = page.get("items", [])
        if not batch:
            break
        titres.extend(batch)
        count += limit #on récupère les sons 0-49 puis 50-99 etc
    return titres
