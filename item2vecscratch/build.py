from collections import Counter
import numpy as np
from loading import load500
# TODO: description fonction, regarder dataset en détail
def build_vocab(playlists, min_count=3):
    counter = Counter()

    for pl in playlists:
        counter.update(pl)
    filtre = [track for track, c in counter.items() if c >= min_count] #si les musiques apparaissent - que min_count on les vire
    track2id = {item: idx for idx, item in enumerate(filtre)}

    id2track = filtre
    counts = np.array([counter[item] for item in id2track], dtype=np.float32)
    p_filtre= [
        [track for track in pl if track in filtre]
        for pl in playlists
    ]
    return track2id, id2track, counts, p_filtre

def build_noise(count,alpha=0.75):
    noise= count**alpha
    noise_dist=noise/noise.sum()
    return noise_dist


def gen_couples1(p,t2id,wdow):
    pairs=[]
    for pl in p:
        for i in range(len(pl)):
            for p1 in range(1,min(wdow,i)+1):

                pairs.append((t2id[pl[i]],t2id[pl[(i-p1)]]))

            for p2 in range(1,min(wdow,len(pl)-i-1) +1):

                pairs.append((t2id[pl[i]],t2id[pl[i+p2]]))
    return pairs

def gen_couples(p,t2id,wdow):
    pairs=[]
    for pl in p:
        for i in range(len(pl)):
            centre=t2id[pl[i]]

            #fenetre <-
            for k in range(1,wdow+1):
                p1=i-k
                if i-k<0:
                    break
                pairs.append((centre,t2id[pl[p1]]))
            
            #fenetre ->
            for l in range(1,wdow+1):
                p2=i+l
                if p2>=len(pl):
                    break
                pairs.append((centre,t2id[pl[p2]]))
    return pairs

p=load500('C:\\Users\\damie\Documents\\MusicRecommender\\spotify_dataset.csv')
track2id, id2track, counts,pl=build_vocab(p,3)
print(gen_couples(pl,track2id,2))