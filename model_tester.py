import keras
import numpy as np

from audio_preprocessing import generate_spectrogram

MODEL_PATH = 'models/model_playfulness.keras'
AUDIO_FILE_PATH = 'data/audioset_audios/0CYxiRkfVhw_cut.mp3'

# load model
model_playfulness = keras.models.load_model(MODEL_PATH)

# audio preprocessing
spectrogram = generate_spectrogram(AUDIO_FILE_PATH)

# predict
y = np.array([spectrogram])
prediction_playfulness = model_playfulness.predict(y)

print('predicted playfulness with {:.3f}'.format(prediction_playfulness[0][0]))



