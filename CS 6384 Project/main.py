import cv2
import os
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


# feature extraction
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping invalid image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (50, 50))

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    return features


# load training data
def load_dataset(base_path):
    data = []
    labels = []
    emotions = ['happy', 'sad', 'angry']

    for label, emotion in enumerate(emotions):
        folder = os.path.join(base_path, emotion)
        for file in os.listdir(folder):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            path = os.path.join(folder, file)
            features = extract_features(path)
            if features is not None:
                data.append(features)
                labels.append(label)

    return np.array(data), np.array(labels)


# train SVM
X_train, y_train = load_dataset('training images')

model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("Training complete")


# predict image
def predict_image(image_path, model):
    features = extract_features(image_path)
    if features is None:
        return None

    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction


# evaluate test image
def evaluate_image(test_folder, model):
    y_true = []
    y_pred = []
    emotions = ['happy', 'sad', 'angry']

    print("\nPredictions")
    for file in os.listdir(test_folder):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(test_folder, file)

        # extract true label from filename
        filename_lower = file.lower()
        true_emotion = None
        for idx, emotion in enumerate(emotions):
            if filename_lower.startswith(emotion):
                true_emotion = emotion
                true_label = idx
                break

        if true_emotion is None:
            print(f"Skipping {file}: cannot determine true emotion from filename")
            continue

        # predict emotion
        prediction = predict_image(path, model)
        if prediction is None:
            continue

        # print prediction
        print(f"{file}: Predicted = {emotions[prediction]}, True = {true_emotion}")

        y_true.append(true_label)
        y_pred.append(prediction)

    # accuracy percentage
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\nModel Accuracy Percentage: {acc:.2f}%")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=emotions, columns=emotions)
    print("\nConfusion Matrix:")
    print(df_cm)


# run evaluation
evaluate_image('test images', model)
