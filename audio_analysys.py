import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ---- Paramètres (à ajuster selon le niveau de détail voulu)
audio_path = "BlindingLights.mp3"
sr_target   = None        # None = garder le sr d'origine (sinon, ex: 22050)
n_fft       = 2048        # taille de fenêtre FFT
hop_length  = 512         # pas entre fenêtres (samples)
win         = "hann"      # fenêtre
# ----

# Charger l'audio (mono par défaut; mettre mono=False pour garder la stéréo)
y, sr = librosa.load(audio_path, sr=sr_target, mono=True)

# STFT -> magnitude en dB
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=win))
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Affichage
plt.figure(figsize=(12, 5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='log')  # y_axis='log' = échelle fréquentielle log
plt.colorbar(format="%+2.0f dB", label="Amplitude (dB)")
plt.title("Spectrogramme (STFT) - échelle log")
plt.tight_layout()
plt.show()
