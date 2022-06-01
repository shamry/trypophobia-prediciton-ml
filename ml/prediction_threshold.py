import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing import image
import cv2

model = load_model('../ml_model/trypophobia.h5')

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

img = cv.imread('../input/train/trypo/0af1da7b17bcb3dded9599b516f33b12.png', 0)
img = cv.medianBlur(img, 5)
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
# plt.show()

im = image.load_img(images)
x = image.img_to_array(im)
x = x / 255.
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
ims = np.vstack([x])
classes = model.predict(ims, batch_size=256)

result = classes[0][0]
msg = ""
if result > 0.70:
    msg = "High"
elif result > 0.50:
    msg = "look like trypo"
else:
    msg = "Low"

print(msg,result)
