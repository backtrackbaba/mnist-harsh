import os
import piexif
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sys

def preprocess_images(root, train=True):
    features = []
    labels = []
    
    for subdir, dirs, files in os.walk(root):
        count  = 0
        for file in files:
            tot_files = len(files)
            if not file == ".DS_Store":
                count += 1
                sys.stdout.write("\rFile = " + file + " ----- Progress: {:2.1f}%".format(100 * count/float(tot_files)))
                img = os.path.join(subdir, file)
                if os.stat(img).st_size > 0:
                    piexif.remove(img)
                else:
                    continue
                im = cv2.imread(img)
                im = cv2.resize(im, (32, 32))
                feature = np.array(im, dtype=np.float32)
                features.append(feature)
                # One Hot Encoding
                if train == True:
                    label = os.path.basename(subdir)
                    if label == "1":
                        label = [1,0,0]
                    elif label == "2":
                        label = [0,1,0]
                    else:
                        label = [0,0,1]  
                    labels.append(label)
                else:
                    label = os.path.basename(img)
                    labels.append(label)
                sys.stdout.flush()
    
    if train == True:
        labels = np.array(labels, np.uint8)
        
    features = np.array(features, np.float32) / 255.
    
    return features, labels
	
X, y = preprocess_images("train/", train=True)

X_test, y_test = preprocess_images("test/", train=False)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=0)

np.savez('data', X_train, y_train, X_valid, y_valid, X_test, y_test)

#Model
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01, momentum=0.01, decay=0.0),
              metrics=['accuracy'])

with np.load('data.npz') as data:
    xtrain = data['arr_0']
    ytrain = data['arr_1']
    xval = data['arr_2']
    yval = data['arr_3']
    xtest = data['arr_4']
    ytest = data['arr_5']

slice_size = len(xtrain)
hist = model.fit(xtrain[:slice_size], ytrain[:slice_size],
                    batch_size=128,
                    epochs=125,
                    verbose=0,
                    validation_data=(xval[:slice_size], yval[:slice_size]))
model.save_weights('weights.h5')

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

model.load_weights('weights.h5')

y_pred = model.predict(xtest, batch_size=32, verbose=0)
y_pred = np.argmax(y_pred, axis=1)+1
temp = np.column_stack((ytest,y_pred))
df = pd.DataFrame(temp, columns=['image_name','Type'])
df.to_csv('predictions.csv',index=False)