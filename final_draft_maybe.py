import cv2
import sys
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import time

# Create directories to save captured images and features
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')
if not os.path.exists('features'):
    os.makedirs('features')

def capture_images(name):
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 10:  # Capture 10 images
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            filename = f'captured_images/{name}_{count}.png'
            cv2.imwrite(filename, face)
            count += 1
        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    return image.flatten()

def prepare_data():
    features = []
    labels = []
    for filename in os.listdir('captured_images'):
        if filename.endswith('.png'):
            label = filename.split('_')[0]
            image_path = os.path.join('captured_images', filename)
            feature = extract_features(image_path)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

def save_features(X, y):
    with open('features/features.pkl', 'wb') as f:
        pickle.dump((X, y), f)

def load_features():
    if os.path.exists('features/features.pkl'):
        with open('features/features.pkl', 'rb') as f:
            return pickle.load(f)
    return np.array([]), np.array([])

def train_model():
    X, y = prepare_data()
    if X.size == 0:
        X, y = load_features()
        if X.size == 0:
            print("No features found. Train the model with some data.")
            return None, None
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    clf = SVC(kernel='rbf', gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
    
    save_features(X, y)
    return clf, le

def get_haar_cascade_path():
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the file is in the same directory
        base_path = sys._MEIPASS
    else:
        # When running normally
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, 'cv2', 'data', 'haarcascade_frontalface_default.xml')

def recognize_face(clf, le):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    last_name = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)
            prediction = clf.predict(face_resized)
            name = le.inverse_transform(prediction)[0]
            last_name = name
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Close the camera after 10 seconds
        if time.time() - start_time > 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Mark attendance with the last detected name
    if last_name:
        df = pd.read_excel('attendance.xlsx') if os.path.exists('attendance.xlsx') else pd.DataFrame(columns=['Name', 'Date'])
        now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        if last_name not in df['Name'].values:
            new_entry = pd.DataFrame({'Name': [last_name], 'Date': [now]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_excel('attendance.xlsx', index=False)

if __name__ == "__main__":
    action = input("Enter 'capture' to capture new images or 'record' to take attendance: ").strip().lower()
    if action == 'capture':
        name = input("Enter your name for capturing images: ")
        capture_images(name)
    elif action == 'record':
        clf, le = train_model()
        if clf and le:
            recognize_face(clf, le)
    else:
        print("Invalid action. Please enter 'capture' or 'record'.")