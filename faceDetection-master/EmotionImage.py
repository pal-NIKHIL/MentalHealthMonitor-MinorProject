import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

try:
    # List of image paths
    image_paths = ['./woman.jpg', './child1.webp', './child2.jpg']

    # Create a figure with subplots for each image and emotion chart
    fig, axs = plt.subplots(len(image_paths), 2, figsize=(12, 6 * len(image_paths)))

    for idx, image_path in enumerate(image_paths):
        # Load the image
        img = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if img is None:
            raise FileNotFoundError("Error: Unable to load the image.")

        # Find the face using the cascade classifier
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Analyze emotion using DeepFace
        obj = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = obj[0]['dominant_emotion']

        # Add the detected emotion text to the image
        emotion_text = 'Dominant Emotion: ' + dominant_emotion.capitalize()
        cv2.putText(img, emotion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Plot the image in the left subplot
        axs[idx, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[idx, 0].axis('off')

        # Visualize the emotions using a line chart in the right subplot
        emotions = list(obj[0]['emotion'].keys())
        values = list(obj[0]['emotion'].values())
        axs[idx, 1].plot(emotions, values, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
        axs[idx, 1].set_xlabel('Emotion')
        axs[idx, 1].set_ylabel('Probability')
        axs[idx, 1].set_title('Detected Emotions')

        # Rotate x-axis labels for better visibility
        plt.setp(axs[idx, 1].xaxis.get_majorticklabels(), rotation=45)

    # Show the composite figure
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Error:", e)
