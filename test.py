from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk,Image

l = ["nice","nice1"]

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