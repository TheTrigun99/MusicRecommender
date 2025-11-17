import torch.nn.functional as F
import torch
obj = torch.load("item2vec.pt")
embeddings = obj["embeddings"]          # shape: (nb_tracks, emb_dim)
track2id = obj["track_to_id"]

def nearest(track, topk=10):
    idx = track2id[track]
    norm = F.normalize(embeddings, dim=1)
    scores = torch.matmul(norm, norm[idx])
    best = torch.topk(scores, topk + 1).indices.tolist()

    neighbors = []
    for i in best:
        name = obj["id_to_track"][i]
        if name == track:
            continue
        neighbors.append((name, float(scores[i])))
        if len(neighbors) == topk:
            break
    return neighbors

for name, score in nearest('I Would Die 4 U', 10):
    print(f"{name}: {score:.4f}")

    