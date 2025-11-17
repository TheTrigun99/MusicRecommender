from gensim.models import Word2Vec
from loading import load500
from build import build_vocab
p=load500('C:\\Users\\damie\Documents\\MusicRecommender\\spotify_dataset.csv')
track2id, id2track, counts,pl=build_vocab(p,3)
model = Word2Vec(
    sentences=pl,
    vector_size=128,
    window=5,
    sg=1,            # skip-gram
    negative=10,
    min_count=1,
    workers=4
)

print(model.wv.most_similar("I Would Die 4 U", topn=5))