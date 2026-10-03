import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "deepfer_mobilenetv2.keras"
IMG_SIZE = (96, 96)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

model = tf.keras.models.load_model(MODEL_PATH)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, IMG_SIZE).astype("float32") / 255.0
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

        idx = int(np.argmax(pred))
        label = CLASS_NAMES[idx]
        confidence = float(pred[idx]) * 100

        text = f"{label}: {confidence:.1f}%"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(
            frame, text, (x, max(25, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    cv2.imshow("DeepFER - Real-Time Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
