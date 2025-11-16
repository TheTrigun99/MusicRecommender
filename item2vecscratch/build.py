from collections import Counter
import numpy as np
def build_vocab(playlists, min_count=1):
    counter = Counter()

    for pl in playlists:
        counter.update(pl)
    filtre = [track for track, c in counter.items() if c >= min_count] #si les musiques apparaissent - que min_count on les vire
    track2id = {item: idx for idx, item in enumerate(filtre)}

    id2track = filtre
    counts = np.array([counter[item] for item in id2track], dtype=np.float32)

    return track2id, id2track, counts

def build_noise(count,alpha=0.75):
    noise= count**alpha
    noise_dist=noise/noise.sum()
    return noise_dist

def gen_couples(p,t2id,window):

    return pairs