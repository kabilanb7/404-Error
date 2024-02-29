from keras.models import load_model
import cv2
import numpy as np

# Load pre-trained Xception model
model_path = r'C:\Users\91978\Downloads\Stress-Detection-master\_mini_XCEPTION.102-0.66.hdf5'
model = load_model(model_path)

# Define emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness']

# Open a video capture object
cap = cv2.VideoCapture(0)

def predict_emotion(frame):
    # Preprocess the frame (resize, convert to grayscale, normalize)
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=3)
    face = face / 255.0

    emotion = model.predict(face)
    emotion_label = emotion_labels[np.argmax(emotion)]

    return emotion_label

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion_label = predict_emotion(frame)

    if emotion_label == 'sadness':
        cv2.putText(frame, emotion_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
