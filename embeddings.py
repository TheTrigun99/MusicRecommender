import librosa
import openl3
audio, sr = librosa.load("BlindingLights.mp3", sr=None, mono=True)  # ↓ sr=16 kHz = moins d’images
emb, ts = openl3.get_audio_embedding(
    audio, sr,
    content_type="music",
    input_repr="mel128",       # plus léger que mel256
    embedding_size=512,        # tu l’as déjà
    hop_size=1.0,              # ↑ moins de pas temporels (0.1 par défaut → 10x plus lourd)
    center=False,              # évite du padding inutile
    batch_size=4               # ↓ charge VRAM (essaie 1 si ça OOM encore)
)