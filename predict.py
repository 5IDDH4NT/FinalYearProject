import numpy as np
import cv2
import glob
from keras.preprocessing.image import load_img
from keras.models import load_model

label_list = ['Stop Navigation',
              'Excuse me',
              'I am sorry',
              'Thanks you',
              'Good bye',
              'I love this game',
              'Nice to meet you',
              'You are welcome',
              'How are you',
              'Have a good time']


index = []

# image size
img_size = 224

# Load Saved Model
model = load_model('my_model_final.h5')


# img_size = 244
# image = load_img()
# img = np.array(image)
# img = img / 255.0
# img = img.reshape(1, img_size, img_size, 3)
# label = model.predict(img)
# print(label)


# Loop Through image folder
for img in glob.glob(r"D:\code-lip-reading\output\*.jpg"):
    image = load_img(img, target_size=(img_size, img_size))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, img_size, img_size, 3)
    label = model.predict(img)
    classes_x = np.argmax(label, axis=1)[0]
    index.append(classes_x)


class_ = max(index)
print("The Phase is: ", label_list[class_])








