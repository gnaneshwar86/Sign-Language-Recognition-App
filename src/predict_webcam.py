import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model("models/asl_model.h5")
labels = sorted(os.listdir("dataset/frames"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = labels[np.argmax(prediction)]

    cv2.putText(
        frame, label, (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 0), 2
    )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
