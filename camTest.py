import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

cnn_model = tf.keras.models.load_model("cnn_model.h5")


print("------------------------------------------------------------")
print("Welcome to the Sign Language Interface!")

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("test")
img_name = "opencv_testpic.png"

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
cam.set(cv2.CAP_PROP_FOCUS, 500)


print("Press SPACE to register signs, and ESCAPE to exit.")
print("Message: ", end="")

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27: # ESC pressed
        print("")
        print("Program Ended")
        break
    elif k%256 == 32: # SPACE pressed
        cv2.imwrite(img_name, frame)
        img = Image.open(img_name)
        img = np.asarray(img)
        img = cv2.resize(img,(128,128))
        img = img.reshape(1, 128, 128,3)
        output = cnn_model.predict(img)[0]
        sign = class_names[np.argmax(output)]
        print(sign, end="")

cam.release()
cv2.destroyAllWindows()
