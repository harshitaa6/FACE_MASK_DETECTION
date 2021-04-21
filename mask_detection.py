import tensorflow as tf
import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2
import os
import time
source=cv2.VideoCapture(0)
model = tf.keras.models.load_model("mask_detector.model")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
while True:

    ret, img = source.read()

    if ret == True:
        time.sleep(1 / 25)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 8)

        for (x, y, w, h) in faces:

            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            result = model.predict(face)

            label = np.argmax(result, axis=1)[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display
        cv2.imshow("Face Mask Detection", img)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

source.release()
cv2.destroyAllWindows()