from tensorflow.keras.layers import Input, Lambda,Dropout, Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import os
#from glob import glob

INTI_LR=1e-4
EPOCHS=20
BS=35

DIRECTORY=r"C:\Users\harsh\Documents\MASK DETECT\dataset"
CATEGORIES=['with_mask','without_mask','improper_mask']

data=[]
labels=[]

for category in CATEGORIES:
    path=os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        image=load_img(img_path, target_size=(224,224))
        image=img_to_array(image)
        image=preprocess_input(image)

        data.append(image)
        labels.append(category)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data=np.array(data, dtype="float32")
labels=np.array(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=20, stratify=labels, random_state=42)

aug=ImageDataGenerator(rotation_range=20,
                               zoom_range=0.15,
                               shear_range=0.15,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               horizontal_flip=True,
                               fill_mode="nearest")
IMAGE_SIZE = [224, 224]
baseModel=MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + [3])

headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='flatten')(headModel)
headModel=Dense(128, activation="softmax")(headModel)
headModel=Dropout(0.50)(headModel)
headModel=Dense(3,activation="softmax")(headModel)

model= Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable=False

opt=Adam(lr=INTI_LR, decay=INTI_LR/EPOCHS)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

H=model.fit(aug.flow(trainX,trainY,batch_size=BS),
            steps_per_epoch=len(trainX)//BS,
            validation_data=(testX, testY),
            validation_steps=len(testX)//BS,
            epochs=EPOCHS)

predicts=model.predict(testX,batch_size=BS)

predicts=np.argmax(predicts,axis=1)

print(classification_report(testY.argmax(axis=1), predicts,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
import matplotlib.pyplot as plt
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


