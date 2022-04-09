import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

cnn_model = tf.keras.models.load_model('my_model.h5')

test = pd.read_csv("sign_mnist_test.csv")
test_set = np.array(test, dtype='float32')
X_test = test_set[:, 1:] / 255
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

predict_x = cnn_model.predict(X_test) 
predicted_classes = list(np.argmax(predict_x,axis=1))
predicted_classes = [class_names[x] for x in predicted_classes]

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_name = "opencv_testpic.png"

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE pressed
        cv2.imwrite(img_name, frame)
        cam.release()
        cv2.destroyAllWindows()
        break

from PIL import Image
# Preprocess image for model
img = Image.open(img_name)
img = img.resize((28, 28))
img = np.asarray(img)
img = img[:,:,0] # convert to grayscale
img = np.reshape(img, (28, 28))
img = np.array(img.flatten())
img = img.reshape(1, 28, 28, 1)
output = cnn_model.predict(img)[0]
plt.imshow(img.reshape(28,28))
plt.show()
print("Predicted as:", class_names[np.argmax(output)])
