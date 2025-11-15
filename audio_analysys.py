import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_path = "BlindingLights.mp3"
sr_target   = None        
n_fft       = 2048        # taille de fenêtre FFT
hop_length  = 512         # pas entre fenêtres (samples)
win         = "hann"      # fenêtre


# Charger l'audio (mono par défaut; mettre mono=False pour garder la stéréo)
y, sr = librosa.load(audio_path, sr=sr_target, mono=True)

# STFT -> magnitude en dB
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=win))
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Affichage
plt.figure(figsize=(12, 5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')  # y_axis='log' = échelle fréquentielle log
plt.colorbar(format="%+2.0f dB", label="Amplitude (dB)")
plt.title("Spectrogramme (STFT) - échelle log")
plt.tight_layout()
plt.show()
def audio_features(m):
    "Prend un audio_path et renvoit un dictionnaire de features liés à la musique,"
    "on essaye "
    d={}
    sr_target   = None        
    n_fft       = 2048        # taille de fenêtre FFT
    hop_length  = 512         # pas entre fenêtres (samples)
    win         = "hann"   
    y, sr = librosa.load(m, sr=sr_target, mono=True) #load la musique et chope le sample rate de la musique
    Spectro = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=win)) #utile car on va économiser beaucoup de calculs avec
    chr= librosa.feature.chroma_stft(S=Spectro,sr=sr,n_fft=n_fft,hop_length=hop_length,window=win) #chromagramme calculé qu'une fois
    d['bpm']=librosa.feature.tempo(sr=sr,S=Spectro) #bpm
    tonnetz=librosa.feature.tonnetz(chroma=chr,sr=sr) #frame by frame 
    d['tonnetz']= np.mean(tonnetz, axis=1) #on moyenne sur le temps
    d['energy']=np.mean(librosa.feature.rms(S=Spectro,hop_length=hop_length,frame_length=n_fft))

    return d

    