import os

from pytube import YouTube
from pydub import AudioSegment
import pandas as pd

# Replace 'your_csv_file_path.csv' with the actual path to your CSV file
csv_file_path = 'data/dataset.csv'
# Read the CSV file into a DataFrame, skipping lines starting with '#'
df = pd.read_csv(csv_file_path)

# Display the DataFrame
#print(df)
#print(df.columns)
df['start'] = df['start'].astype(int)
df['stop'] = df['stop'].astype(int)
#print(df.dtypes)
output_path = 'data/audioset_audios'

def download_audio(row, output_path):
    video_id = row['ytid']

    try:
        # Video erhalten
        video = YouTube(f'https://www.youtube.com/watch?v={video_id}')

        # Audio-Stream auswählen
        audio_stream = video.streams.filter(only_audio=True).first()

        # Video herunterladen und schneiden
        audio_stream.download(output_path, filename=f"{video_id}_cut.mp3")

    except Exception as e:
        print(f"Fehler beim Herunterlanden von Video {video_id}: {e}")

def cut_audio(row, output_path):
    video_id = row['ytid']
    start_time = int(row['start'])
    end_time = int(row['stop'])

    try:
        audio_file_path = f"{output_path}/{video_id}_cut.mp3"

        # Audio öffnen und schneiden
        audio = AudioSegment.from_file(audio_file_path)
        cut_audio = audio[start_time * 1000:end_time * 1000]

        # Geschnittene Audio speichern
        cut_audio.export(f"{output_path}/{video_id}_cut.mp3", format="mp3")

    except Exception as e:
        print(f"Fehler beim Verarbeiten von Video {video_id}: {e}")


def process_audio(row, output_path):

    if os.path.isfile(f"{output_path}/{row['ytid']}_cut.mp3"):
        print(f"video {row['ytid']} already downloaded, skipping.")
        return None

    download_audio(row, output_path)
    cut_audio(row, output_path)

# download & cut audios
df.apply(lambda row: process_audio(row, output_path), axis=1)
