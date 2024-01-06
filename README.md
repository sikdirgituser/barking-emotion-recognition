# Regocnise dog emotions from barking

Project in Coin-Seminar 2023/24

## setup    

Make sure you have python installed on your workstation. Then run the following commands in terminal/shell:

```console
# create virtual environment
python3 -m venv venv

# activate virtualenv
source venv/bin/activate

# install requirements from textfile
pip install -r requirements.txt

```

If you need other packages, make sure to add them to the requirements.txt file with `pip freeze > requirements.txt
`

## files

### preprocessing

- _dataset_preprocessing.py_: Download current state of labels from sheets, compare and write identic labels to `data/dataset.csv`
- _audioset_download.py_: Download and cut the corresponding audio files, save to `data/audioset_audios/`
- _audio_preprocessing.py_: Helper functions to generate (mel-)spectrograms of same shape from audio files

### modelling

- _model_playfulness.py_: first approach of a model predicting if barking is playful or not.
- ...

### deployment

- Models should be saved to `/models`, presumably as `.keras`-file
- _server.py_: boilerplate to deploy the model(s) into a simple API through flask