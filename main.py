import cv2
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
from tkinter import messagebox as tk1
from keras.models import model_from_json
from tkinter.filedialog import askopenfile


class Application:
    def __init__(self, output_path = "./"):

        json_file=open(r"models/dark_s.json","r")
        loaded_model_json=json_file.read()
        json_file.close()
        self.loaded_model=model_from_json(loaded_model_json)
        self.loaded_model.load_weights("models/dark_s.h5")

        json_file=open(r"models/eyes_d.json","r")
        loaded_model_json=json_file.read()
        json_file.close()
        self.loaded_model1=model_from_json(loaded_model_json)
        self.loaded_model1.load_weights("models/eyes_d.h5")

        json_file=open(r"models/wrinkl.json","r")
        loaded_model_json=json_file.read()
        json_file.close()
        self.loaded_model2=model_from_json(loaded_model_json)
        self.loaded_model2.load_weights("models/wrinkl.h5")

        self.root = tk.Tk()
        self.root.title("Aging Signs Detection")

        self.panel = ttk.Label(self.root)
        self.panel.pack(padx=10, pady=10)
        self.Imgarr = tk.StringVar()

        btn = ttk.Button(self.root, text="Upload Image", command=self.upload)
        btn2 = ttk.Button(self.root, text="Predict", command= self.predict)
        btn1 = ttk.Label(self.root,text = "Upload the Image : ",width=20,font=("Consolas",9))
        
        btn1.pack(side = "left", expand = True, fill = "both")
        btn.pack(side = "left", expand = True, fill = "both")
        btn2.pack(side = "left", expand = True, fill = "both")

        cv2image = cv2.imread('images\\nice.jpg')
        self.current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=self.current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

    def upload(self):
        a = askopenfile(parent=self.root,initialdir='test_images/',initialfile='nice')
        try:
            a = a.name
            if a != None:
                self.Imgarr.set(a)
                tk1.showinfo("Sucessful","The Image uploaded Sucessfully")
            else :
                tk1.showerror("Error","The Image is not uploaded Sucessfully")
        except:
            tk1.showerror("Error","The Image is not uploaded Sucessfully")

    def predict(self):
        try:
            l = []
            img1 = cv2.imread(str(self.Imgarr.get()))
            img = cv2.resize(img1,(50,50))
            img = img.reshape(-1, 50, 50, 3)
            result = self.loaded_model.predict(img)
            result = result[0]
            if result[0] >= result[1]:
                l.append("dark spots")
            else:
                l.append("no dark spots")

            result = self.loaded_model1.predict(img)
            result = result[0]
            if result[0] >= result[1]:
                l.append("no puffy eyes")
            else:
                l.append("puffy eyes")

            result = self.loaded_model2.predict(img)
            result = result[0]
            if result[0] >= result[1]:
                l.append("no wrinkles on face")
            else:
                l.append("wrinkles on face")

            current_image = Image.fromarray(img1)
            imgtk = ImageTk.PhotoImage(image=current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            tk1.showinfo("Sucessful","Predictions are : "+str(l))
        
        except:
            tk1.showerror("Error","The Image is not uploaded Sucessfully")


pba = Application('images\\nice\\')
pba.root.mainloop()