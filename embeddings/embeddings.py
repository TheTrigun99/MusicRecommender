import librosa
import openl3

model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
                                                 embedding_size=512)

