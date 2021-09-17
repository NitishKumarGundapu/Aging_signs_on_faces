import cv2
from tkinter.ttk import *
from tkinter import *
import warnings
from PIL import ImageTk,Image
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


Imgarr0 = cv2.imread(r"test_images/56.jpg")
Imgarr1=cv2.resize(Imgarr0,(50,50))
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

rootg = Tk()
rootg.resizable(False,False)
rootg.geometry('450x450')
rootg.title('Facial Predictions')
u2 = Image.open("images/images_2.jpg")
u2 = u2.resize((450,450),Image.ANTIALIAS)
u2 = ImageTk.PhotoImage(u2)
Label(rootg,image=u2).place(x=0, y=0, width=450, height=450)

img = cv2.resize(Imgarr0, (300, 250))
im = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=im) 
Label(rootg, image=imgtk).place(x=70,y=50)

Label(rootg, text = "Predictions are : "+str(l)).place(x=50,y=350)
rootg.mainloop()