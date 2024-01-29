import os

from pytube import YouTube
from pydub import AudioSegment
import pandas as pd

# read in the dataset
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

    if os.path.isfile(f"{output_path}/{row['ytid']}.mp3"):
        print(f"video {row['ytid']} already downloaded, skipping.")
        return None

    try:
        # Video erhalten
        video = YouTube(f'https://www.youtube.com/watch?v={video_id}')

        # Audio-Stream auswählen
        audio_stream = video.streams.filter(only_audio=True).first()

        print(f"download {video_id}")

        # Video herunterladen
        audio_stream.download(output_path, filename=f"{video_id}.mp3")

    except Exception as e:
        print(f"Fehler beim Herunterladen von Video {video_id}: {e}")

def cut_audio(row, output_path, full_audio_path):
    video_id = row['ytid']
    start_time = int(row['start'])
    end_time = int(row['stop'])

    export_path = f"{output_path}/{video_id}_{start_time}_{end_time}_cut.mp3"

    if os.path.isfile(export_path):
        print(f"video {export_path} already cut, skipping.")
        return None

    try:
        audio_file_path = f"{full_audio_path}/{video_id}.mp3"

        # Audio öffnen und schneiden
        audio = AudioSegment.from_file(audio_file_path)
        cut_audio = audio[start_time * 1000:end_time * 1000]

        # Geschnittene Audio speichern
        cut_audio.export(export_path, format="mp3")

    except Exception as e:
        print(f"Fehler beim Verarbeiten von Video {video_id}: {e}")


def process_audio(row, output_path):

    download_audio(row, f"{output_path}/full_video")
    cut_audio(row, output_path, f"{output_path}/full_video")


if __name__ == "__main__":
    # download & cut audios
    df.apply(lambda row: process_audio(row, output_path), axis=1)
