import os
import cv2
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout

features,target = [],[]

for x in ['nodarkspots', 'darkspots']:
    ImagesNamesList=os.listdir("images//train3" + "/" + str(x))
    for y in ImagesNamesList:
        Imgarr=cv2.imread(r"images//train3" + "/" + str(x) + "/" + y)
        try:
            Imgarr=cv2.resize(Imgarr,(50,50))
            features.append(Imgarr)
        except:
            pass
        else:
            if x=="nodarkspots":
                target.append(0)
            else:
                target.append(1)
    print("In Folder", x)

features = np.array(features)
target = np.array(target)

#features = tf.expand_dims(features, axis=-1)

features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2)

print(features_train.shape)
print(target_train.shape)
print(features_test.shape)
print(target_test.shape)

dataGen=ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1)
batches=dataGen.flow(features_train,target_train,batch_size=20)
images,labels=next(batches)

target_train=to_categorical(target_train)

model=Sequential()
model.add(Conv2D(100,(3,3),activation="relu",input_shape=(50,50,3)))
model.add(Conv2D(200,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(100,(3,3),activation="relu"))
model.add(Conv2D(100,(3,3),activation="relu"))
model.add(Conv2D(100,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(dataGen.flow(features_train,target_train,batch_size=20),epochs=20)