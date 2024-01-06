from flask import Flask, request, jsonify
from flask_cors import CORS

from audio_preprocessing import generate_spectrogram

app = Flask(__name__)
CORS(app)

MODEL_PATH_PLAYFULNESS = 'models/model_playfulness.keras'

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        audio_file = request.files['audio']

        # Save the audio file locally as a WAV file
        audio_path = 'received_audio.mp3'
        audio_file.save(audio_path)

        # with wave.open(audio_path, 'wb') as wf:
        #     wf.setnchannels(1)  # mono
        #     wf.setsampwidth(2)  # 16-bit
        #     wf.setframerate(44100)  # sample rate (adjust accordingly)
        #     wf.writeframes(audio_file.read())

        # generate spectrogram from audio file
        # spectrogram = generate_spectrogram(audio_path)

        # load models
        # model_playfulness = keras.models.load_model(MODEL_PATH_PLAYFULNESS)

        # predict
        # y = np.array([spectrogram])
        # prediction_playfulness = model_playfulness.predict(y)
        # ... other models

        # prepare output
        model_output = [
            {"name": "playful",
             "prob": 0},
            {"name": "happy",
             "prob": 0},
            {"name": "fearful",
             "prob": 0},
            {"name": "aggressive",
             "prob": 0},
        ]

        # For this example, just returning a success message
        return jsonify(
            {'message': 'Audio processed successfully', 'result': model_output})

    except Exception as e:
        # Log the error and return an error message
        print(f'Error processing audio: {str(e)}')
        return jsonify({'error': 'Error processing audio'}), 500


if __name__ == '__main__':
    app.run(debug=True)
