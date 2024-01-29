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
from audio_preprocessing import generate_spectrogram, random_frequency_mask
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# read dataset
df = pd.read_csv('data/dataset.csv')

# choose what to include in the model
df = df[df.label.isin(['Playfulness', 'Aggressiveness'])]

# Apply feature extraction to each audio file
df['spectrogram'] = df.apply(lambda row: generate_spectrogram(f"data/audioset_audios/{row['ytid']}_{int(row['start'])}_{int(row['stop'])}_cut.mp3"), axis=1)

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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# output data-dimensions
print(f"train model on {X_train.shape[0]} audio-files, test on {X_test.shape[0]}.")

# generate class weights as we have unbalanced data
y_train_list = y_train.argmax(axis=1).tolist()
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_list), y=y_train_list)
class_weights_dict = dict(enumerate(class_weights))

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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train.reshape((*X_train.shape, 1)), y_train, class_weight=class_weights_dict, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model performance on the test set
accuracy = model.evaluate(X_test.reshape((*X_test.shape, 1)), y_test)[1]
print()
print(f'Accuracy on test-set: {accuracy}')
print()

# make predictions for test data
y_pred = model.predict(X_test.reshape((*X_test.shape, 1))).round()

# Create confusion matrix
conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test[:,0], y_pred[:,0]))

# model export
model.save('models/model_all_categories.keras')