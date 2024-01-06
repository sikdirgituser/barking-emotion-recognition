import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from audio_preprocessing import generate_spectrogram

# read dataset
df = pd.read_csv('data/dataset.csv')

# Apply feature extraction to each audio file
df['spectrogram'] = df['ytid'].apply(lambda ytid: generate_spectrogram(f'data/audioset_audios/{ytid}_cut.mp3'))

# show files not found
print(f"{df['spectrogram'].isna().sum()} entries dropped because no audio file found.")

# Remove rows where spectrogram is None (i.e., file not found)
df = df.dropna(subset=['spectrogram'])

# Encode emotion labels
label_encoder = LabelEncoder()
df['emotion_label_encoded'] = label_encoder.fit_transform(df['label'])

# Create X (features) and y (labels)
X = np.array(df['spectrogram'].tolist())
y = to_categorical(df['emotion_label_encoded'])  # One-hot encode labels
y = to_categorical(np.array(df['label'] == 'Playfulness'))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# output data-dimensions
print(f"train model on {X_train.shape[0]} audio-files, test on {X_test.shape[0]}.")

# Build CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train.reshape((*X_train.shape, 1)), y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model performance on the test set
accuracy = model.evaluate(X_test.reshape((*X_test.shape, 1)), y_test)[1]
print(f'Accuracy: {accuracy}')
print('')

# confusion matrix
y_pred = model.predict(X_test.reshape((*X_test.shape, 1))).round()
result = confusion_matrix(y_test[:,0], y_pred[:,0], normalize='pred')
print('confusion matrix:')
print(result)

# model export
model.save('models/model_playfulness.keras')