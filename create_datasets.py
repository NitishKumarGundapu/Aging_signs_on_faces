import os
import cv2
import numpy as np

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


dark_spot_features = np.array(features)
dark_spot_target = np.array(target)

print(dark_spot_features.shape)
print(dark_spot_target.shape)

np.savez_compressed('datasets/dark_spot_data.npz', dark_spot_features, dark_spot_target)

features,target = [],[]

for x in ['no puffy eyes', 'puffy eyes']:
    ImagesNamesList=os.listdir("images//train2" + "/" + str(x))
    for y in ImagesNamesList:
        Imgarr=cv2.imread(r"images//train2" + "/" + str(x) + "/" + y)
        try:
            Imgarr=cv2.resize(Imgarr,(50,50))
            features.append(Imgarr)
        except:
            pass
        else:
            if x=="no puffy eyes":
                target.append(0)
            else:
                target.append(1)
    print("In Folder", x)


eyes_features = np.array(features)
eyes_target = np.array(target)

print(eyes_features.shape)
print(eyes_target.shape)

np.savez_compressed('datasets/eyes_data.npz', eyes_features, eyes_target)

features,target = [],[]

for x in ['NoWrinkles', 'wrinkles']:
    ImagesNamesList=os.listdir("images//train1" + "/" + str(x))
    for y in ImagesNamesList:
        Imgarr=cv2.imread(r"images//train1" + "/" + str(x) + "/" + y)
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


wrinkles_features = np.array(features)
wrinkles_target = np.array(target)

print(wrinkles_features.shape)
print(wrinkles_target.shape)

np.savez_compressed('datasets/wrinkles_data.npz', wrinkles_features, wrinkles_target)