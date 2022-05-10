from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as k
import tensorflow as tf

model = Sequential()
model.add(Dense(12, input_dim=8,  activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    


with tf.GradientTape() as g:
    outputTensor = model.output

    listOfVariableTensors = model.trainable_weights
dy_dw = g.jacobian(outputTensor, listOfVariableTensors)
print(dy_dw)
#gradients = k.gradients(outputTensor, listOfVariableTensors)
#print(gradients)