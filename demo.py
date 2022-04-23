import PySimpleGUI as sg
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import time
import threading

def run_model(model, class_names, frame, window, target, img_name):
    cv2.imwrite(img_name, frame)
    img = Image.open(img_name)
    img = np.asarray(img)
    img = cv2.resize(img,(128,128))
    img = img.reshape(1, 128, 128,3)
    output = model.predict(img)[0]
    sign = class_names[np.argmax(output)]
    if len(output) == 24 and output[target] > 0.5:
        window['-OUTPUT-'].update(class_names[target])
    else:
        window['-OUTPUT-'].update(sign)

def main():

    phrase = "BEWARE"
    index = 0


    seconds = time.time()
    prev = seconds
    
    sg.theme('Black')

    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]
    vowel_classes = ['A', 'E', 'I', 'O', 'U']

    cnn_model = tf.keras.models.load_model("cnn_model.h5")
    tf_model = tf.keras.models.load_model("TF_model.h5")
    vowel_model = tf.keras.models.load_model("cnn_model_vowel.h5")

    # define the window layout
    layout = [[sg.Text('Sign Language Demo', size=(30, 1), justification='center', font='Helvetica 20', key='prompt')],
            [sg.Text("Try signing: " + phrase, size=(50,1), font='Helvetica 12', key='-PROMPT-')],
            [sg.Column([[sg.Image(filename='', size=(10, 50), key='image')]], justification='center')],
              [sg.Button('Start', size=(10, 1), font='Helvetica 10'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 10'), 
               sg.Text(size=(3,1), font='Helvetica 15', key='-OUTPUT-'),
               sg.Text(size=(30,1), font='Helvetica 10', key='-MSG-')]]

    # create the window and show it without the plot
    window = sg.Window('Sign Language Interface',
                       layout, location=(300, 100))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    recording = False

    while True:

        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        elif event == 'Start':
            recording = True
            index = 0

        if recording:
            window['-PROMPT-'].update("Please sign:" + phrase[index])
            ret, frame = cap.read()
            seconds = time.time()
            if seconds - prev > 0.5:
                prev = seconds
                threading.Thread(target=run_model, args=(cnn_model, class_names, frame, window, 
                class_names.index(phrase[index]), "pic1.png")).start()
                threading.Thread(target=run_model, args=(tf_model, class_names, frame, window, 
                class_names.index(phrase[index]), "pic2.png")).start()
                threading.Thread(target=run_model, args=(vowel_model, vowel_classes, frame, window, 
                class_names.index(phrase[index]), "pic3.png")).start()

            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

            if window['-OUTPUT-'].get() == phrase[index]:
                window['-MSG-'].update("Correct Letter Detected: " + phrase[index])
                index += 1
                if index == len(phrase):
                    recording = False
                    window['-PROMPT-'].update("Congratulations, you completely signed: " + phrase)
                time.sleep(0.5)
             
main()
