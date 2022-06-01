from keras.models import load_model
from keras.preprocessing import image
import numpy as np


model = load_model('ml_model/trypophobia.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def predict(test_image):
    img = image.load_img(test_image, target_size=(128, 128))
    x = image.img_to_array(img)
    x = x / 255.
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    images = np.vstack([x])
    classes = model.predict(images, batch_size=256)

    result = classes[0][0]
    msg = ""
    if result > 0.70:
        msg = "HIGH"
    elif result > 0.50:
        msg = "Medium"
    else:
        msg = "LOW"
    return {
        "info":msg,
        "percentage":str(round(result*100))+"%"
    }
