import pandas as pd
import numpy as np
from keras.src.metrics import F1Score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.svm import SVC

from audio_preprocessing import generate_spectrogram, random_frequency_mask
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# read dataset
df = pd.read_csv('data/dataset.csv')

# remove Neutral
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
y = df['emotion_label_encoded']  # One-hot encode labels

# Flatten the spectrograms
X_flat = X.reshape((X.shape[0], -1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# output data-dimensions
print('------------')
print(f"train model on {X_train.shape[0]} audio-files, test on {X_test.shape[0]}.")

# Initialize the SVM classifier
svm_classifier = SVC()

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.01, 0.1, 1]}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, scoring='accuracy', cv=3, verbose=10)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
