import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy
from PIL import ImageTk, Image

#initialise GUI
top=tk.Tk()
top.geometry('800x800')
top.title('Traffic sign Detection')
top.configure(background='black')

label=Label(top,background='green', font=('Calibri',16,'italic'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((50,50))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    #pred = model.predict_classes([image])[0]
    #sign = classes[pred+1]
    #print(sign)
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=Button(top,text="Detect Sign",command=lambda: classify(file_path),padx=5,pady=9)
    classify_b.configure(background='#000fff', foreground='White',font=('arial',13,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=13,pady=7)
upload.configure(background='#00e6e6', foreground='#000000',font=('arial',15,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)

heading = Label(top, text="Traffic Sign Detection",padx=5, pady=20, font=('Calibri',20,'bold'))
heading.configure(background='#ffff00',foreground='#364156')
heading.pack()
top.mainloop()
