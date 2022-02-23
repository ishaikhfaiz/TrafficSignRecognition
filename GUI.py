import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy
from PIL import ImageTk, Image                            # for image reading
from keras.models import load_model
from gtts import gTTS  # text to speech
import os                                                 # operating system access
import subprocess
import threading
import time
import pyttsx3
import speech_recognition as sr  # Speech Recognition
import datetime
import webbrowser
import os
from newsapi.newsapi_client import NewsApiClient  # Google News-Api


# importing traffic_sign_recognition module
model = load_model('./traffic_sign_train_recognition.h5')

classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'
           }

# initialise GUI
top = tk.Tk()
top.geometry('800x600')  # window size
top.title('Traffic sign Detection')
top.configure(background='black')

label = Label(top, foreground='white', background='black')
sign_image = Label(top)
sign = ''

# Voice Engine
engine = pyttsx3.init("espeak")
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[11].id)


def speak(audio):  # Output Voice
    engine.say(audio)
    engine.runAndWait()

# Take Input


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print("Say that again please...")
        return "None"
    return query

# news api


def news():
    newsapi = NewsApiClient(api_key='5840b303fbf949c9985f0e1016fc1155')
    speak("What topic you need the news about")
    topic = takeCommand()
    data = newsapi.get_top_headlines(
        q=topic, language="en", page_size=5)
    newsData = data["articles"]
    for y in newsData:
        speak(y["description"])

# command processing
def car_assistant():
    while True:
        query = takeCommand().lower()

        if 'open gmail' in query:
            webbrowser.open("gmail.com")
            break

        elif 'play music' in query:
            music_dir = '/home/shaikhfaiz'
            os.system("rhythmbox-client --play")

        elif 'stop music' in query:
            os.system("rhythmbox-client --stop")
            break

        elif 'Next Song' in query:
            os.system("rhythmbox-client --next")

        elif 'previous Song' in query:
            os.system("rhythmbox-client --previous")

        elif 'Increase Volume' in query:
            os.system("rhythmbox-client --volume-up")

        elif 'Decrease Volume' in query:
            os.system("rhythmbox-client --volume-down")
            print("rhythmbox-client --print-volume")

        elif 'Set Volume' in query:
            os.system("rhythmbox-client --set-volume")

        elif 'what is the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M")
            speak("Sir, the time is {strTime}")
            break

        elif 'news' in query:
            news()
            break

# Text to speech class


def speech():             # Output in form of Audio
    myobj = gTTS(text=sign, lang="en")
    myobj.save("sign.mp3")
    # for window operating system uncomment-> os.system("start /.sign.mp3")
    os.system("mpg321 ./sign.mp3")

# Giving Image Input


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = numpy.argmax(model.predict([image])[0])
    global sign
    sign = classes[pred]
    print(sign)
    label.configure(font=('Yusei Magic', 17, 'bold'),
                    text=sign, anchor="center")


# Taking path of image manually
def upload_image():
    try:
        global file_path
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((500, 500))
        #uploaded.resize((1000, 1000), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
    except:
        pass


sign_image.place(relx=0.50, rely=0.45, anchor="center")
label.place(relx=0.52, rely=0.71, anchor="center")

border_color = Frame(top, background="black")

# Button for Speech
speak_btn = PhotoImage(file="Speak.png")
speech_b = Button(top, text='Speak', image=speak_btn,
                  bd=0, command=lambda: speech())
speech_b.place(relx=0.73, rely=0.50)

# Button for Activating Assistant
assis_btn = PhotoImage(file="SmartAssist.png")
assist_b = Button(top, text='Tell Me', image=assis_btn,
                  bd=0, command=lambda: car_assistant())
assist_b.place(relx=0.07, rely=0.50)

# Button for Select
select_btn = PhotoImage(file="Select.png")
upload = Button(top, text="Upload an image", bd=0,
                image=select_btn, command=upload_image)
upload.place(relx=0.23, rely=0.80)

# Button for Detect
detect_btn = PhotoImage(file="Detect.png")
classify_b = Button(top, text="Detect Sign", image=detect_btn,
                    bd=0, command=lambda: classify(file_path))
classify_b.place(relx=0.55, rely=0.80)
classify_b.image = detect_btn

# Heading
heading = Label(top, text="TRAFFIC SIGN DETECTION USING MACHINE LEARNING", wraplength=750,
                justify='center', padx=5, pady=20, font=('Work Sans', 25, 'bold'))  # GUI heading
heading.configure(background='black', foreground='#FEC80A')
heading.pack()
top.mainloop()
