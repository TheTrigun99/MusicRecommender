import os
import librosa
import openl3
import numpy as np
from tqdm import tqdm
import json


def trackembedded(path,model):
    '''
    On charge une  avec un certain modèle Openl3 et on en fait un embedding
    path -> (vecteur,metadonnée)
    '''
    y, sr = librosa.load(path, sr=None, mono=True)
    emb, _ = openl3.get_audio_embedding(
        y, sr,
        model=model,hop_size=1.0, center=False, batch_size=64)
    return emb.mean(axis=0)    

def wholetracks(dossier,model):
    '''
    Charge tout un dossier et renvoie les embeddings
    '''
    X, meta = [], []
    for music in tqdm(os.listdir(dossier)): #on liste les musiques
        if not music.lower().endswith((".mp3", ".wav")): #on va associer titre au nom du vecteur
            continue
        path = os.path.join(dossier, music)
        try:
            vec = trackembedded(path, model)
            X.append(vec)
            meta.append({
                "title": os.path.splitext(music)[0],
                "path": path
            })
        except Exception as e:
            print(f"[WARN] erreur sur {music}: {e}")

    X = np.vstack(X).astype("float32")
    return X, meta    #on construit les 2 en même temps pour pas de soucis d'alignement

model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",embedding_size=512)
X,meta=wholetracks("/mnt/c/Users/damie/Documents/Musiques/old songs300",model)

with open("embeddings_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
np.save("embeddings_library.npy", X)

