import pandas as pd
import numpy as np
from keras.src.metrics import F1Score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from audio_preprocessing import generate_spectrogram, random_frequency_mask, generate_mel_spectrogram

AUGMENTATION_FACTOR = 1

# read dataset
df = pd.read_csv('data/dataset.csv')

# remove Neutral
df = df[df.label.isin(['Happiness', 'Playfulness', 'Aggressiveness', 'Despair', 'Fearfulness'])]

# Apply feature extraction to each audio file
df['spectrogram'] = df.apply(lambda row: generate_spectrogram(f"data/audioset_audios/{row['ytid']}_{int(row['start'])}_{int(row['stop'])}_cut.mp3"), axis=1)

# show files not found
print(f"{df['spectrogram'].isna().sum()} entries dropped because no audio file found.")

# Remove rows where spectrogram is None (i.e., file not found)
df = df.dropna(subset=['spectrogram'])

# Encode emotion labels
label_encoder = LabelEncoder()
df['emotion_label_encoded'] = label_encoder.fit_transform(df['label'])

spectograms = df['spectrogram'].tolist()

# Augment data by random frequency masks, see: https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52
augmented_spectograms = []
for i in range(AUGMENTATION_FACTOR - 1):
    augmented_spectograms = augmented_spectograms + [random_frequency_mask(spec) for spec in spectograms]
all_spectograms = augmented_spectograms + spectograms

# Create X (features) and y (labels)
X = np.array(all_spectograms)

# labels
labels = (df['label'] == 'Playfulness').tolist()

# Repeat the labels 10 times (9 + 1) to make sure we have same amount as augmented data
all_labels = np.tile(labels, AUGMENTATION_FACTOR)

# Convert to one-hot encoding using to_categorical
y = to_categorical(all_labels)

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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', F1Score()])

# Train the model
model.fit(X_train.reshape((*X_train.shape, 1)), y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model performance on the test set
accuracy = model.evaluate(X_test.reshape((*X_test.shape, 1)), y_test)[1]
print()
print(f'Accuracy on test-set: {accuracy}')
print('')

# make predictions for test data
y_pred = model.predict(X_test.reshape((*X_test.shape, 1))).round()

# Confusion Matrix
cm = confusion_matrix(y_test[:,0], y_pred[:,0])
print("Confusion Matrix:")
print(cm)
print()

# Classification Report
print("Classification Report:")
print(classification_report(y_test[:,0], y_pred[:,0]))

# show confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['playful', 'not playful'])
cm_display.plot(cmap='Blues')
plt.show()

# model export
model.save('models/model_playfulness.keras')