import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_values = [0] * len(emotions)
bars = ax.bar(emotions, emotion_values)
ax.set_ylim(0, 100)
ax.set_ylabel('Probability')
ax.set_title('Real-time Emotion Detection')

face_cascade = cv2.CascadeClassifier('faceDetection-master/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    emotion_values = [0] * len(emotions)

    for (x, y, w, h) in faces:
        try:
            result = DeepFace.analyze(img_path=frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]["dominant_emotion"]
            emotion_index = emotions.index(dominant_emotion)
            emotion_values[emotion_index] = result[0]['emotion'][dominant_emotion] * 100
        except Exception as e:
            print("Error:", e)

        txt = f"Emotion: {dominant_emotion}"
        cv2.putText(frame, txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for bar, value in zip(bars, emotion_values):
        bar.set_height(value)
    
    plt.draw()
    plt.pause(0.001)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plt.ioff()
plt.show()
