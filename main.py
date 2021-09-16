import os
import sys
import cv2
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter("ignore")

def preprocessing(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=image/255
    return image

features=[]
target=[]
for x in ['NoWrinkles', 'Wrinkles']:
    ImagesNamesList=os.listdir(r"train" + "/" + str(x) )
    for y in ImagesNamesList:
        Imgarr=cv2.imread(r"train" + "/" + str(x) + "/" + y)
        try:
            Imgarr=cv2.resize(Imgarr,(50,50))
            features.append(Imgarr)
        except:
            pass
        else:
            if x=="NoWrinkles":
                target.append(0)
            else:
                target.append(1)
    print("In Folder", x)


features=np.array(features)
target = np.array(target)

print(features.shape)
print(target.shape)

features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2)

print(features_train.shape)
print(target_train.shape)
print(features_test.shape)
print(target_test.shape)

