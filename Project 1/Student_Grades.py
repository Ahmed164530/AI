import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


zip_path = r'F:\Gmy\projects\Game_Project\New folder\data\archive.zip'
extract_folder = r'F:\Gmy\projects\Game_Project\New folder\data\FER2013'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)


data = pd.read_csv(os.path.join(extract_folder, 'fer2013.csv'))


X = []
y = []

for index, row in data.iterrows():
    emotion = row['emotion']
    pixels = np.fromstring(row['pixels'], sep=' ')
    X.append(pixels)
    y.append(emotion)

X = np.array(X)
y = np.array(y)

X = X.astype('float32') / 255  
X = X.reshape(-1, 48, 48, 1)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=7)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))  


model.save('your_model.h5')
print("تم حفظ النموذج.")

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model


model = load_model('your_model.h5')  


def classify_expression(image):
    image = cv2.resize(image, (48, 48))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction)


def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        expression = classify_expression(gray_image)
        display_image(img, expression)


def display_image(img, expression):
    expressions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    cv2.putText(img, f'Expression: {expressions[expression]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img


root = tk.Tk()
root.title('Facial Expression Recognition')
root.geometry('600x500')


btn_load = tk.Button(root, text='Load Image', command=load_image)
btn_load.pack(pady=20)


panel = tk.Label(root)
panel.pack(pady=10)

root.mainloop()
