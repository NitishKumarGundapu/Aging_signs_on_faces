import cv2
from tkinter.ttk import *
from tkinter import *
import warnings
from keras.models import model_from_json

warnings.filterwarnings("ignore")

json_file=open(r"models/dark_s.json","r")
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("models/dark_s.h5")

json_file=open(r"models/eyes_d.json","r")
loaded_model_json=json_file.read()
json_file.close()
loaded_model1=model_from_json(loaded_model_json)
loaded_model1.load_weights("models/eyes_d.h5")

json_file=open(r"models/wrinkl.json","r")
loaded_model_json=json_file.read()
json_file.close()
loaded_model2=model_from_json(loaded_model_json)
loaded_model2.load_weights("models/wrinkl.h5")

a = "56.jpg"
Imgarr0 = cv2.imread(r"test_images/"+str(a))
Imgarr1 = cv2.resize(Imgarr0,(50,50))
Imgarr = Imgarr1.reshape(-1, 50, 50, 3)

l = []

result = loaded_model.predict(Imgarr)
result = result[0]
if result[0] >= result[1]:
    l.append("dark spots")
else:
    l.append("no dark spots")

result = loaded_model1.predict(Imgarr)
result = result[0]
if result[0] >= result[1]:
    l.append("no puffy eyes")
else:
    l.append("puffy eyes")

result = loaded_model2.predict(Imgarr)
result = result[0]
if result[0] >= result[1]:
    l.append("no wrinkles on face")
else:
    l.append("wrinkles on face")

print("the predictions are : ",l)