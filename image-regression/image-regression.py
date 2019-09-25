# Regression - Using set of inputs, predict real-valued output

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import backend as K

image = [
     [0,130,255],
     [40,170,255],
     [80,210,255]
 ]
 
image = np.array(image)
 
image = np.divide(image, 255.0)
 
image_width, image_height = image.shape
 
print("Image with shape: {}".format(image.shape))
print(image)


x = []
y = []

for i in range(image_height):
    for j in range(image_width):
        x.append([i/image_height, j/image_width])
        
        y.append([image[i][j]])
        
x = np.array(x)
y = np.array(y)

print("\nScaled coordinates (input):")
print(x)

print("\nScaled pixel brightness values (output):")
print(y)


model = Sequential()

model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(x,y, epochs=500)

predicted_image = model.predict(x, verbose=False).reshape(image.shape)
print("\nPredicted Image:")
print(predicted_image)
K.clear_session()