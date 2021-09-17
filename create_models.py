import os
import cv2
import warnings
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout

warnings.simplefilter("ignore")

def preprocessing(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=image/255
    return image

for a in os.listdir('datasets/'):
    data = np.load(r'datasets/'+str(a))
    features , target = data['arr_0'],data['arr_1']

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

    model_json=model.to_json()
    with open("models/"+str(a[:6])+".json","w") as abc:
        abc.write(model_json)
        abc.close
    model.save_weights("models/"+str(a[:6])+".h5")