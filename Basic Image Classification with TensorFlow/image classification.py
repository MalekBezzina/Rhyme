## import tensorflow ##
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("using TF version",tf.__version__)

## import the dataset ##

from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

#shapes of imported  arrays 

print("x_train shape",x_train.shape)
print("y_train shape",y_train.shape)
print("x_test shape",x_test.shape)
print("y_test shape",y_test.shape)

#plot an image  example 

from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[1],cmap='binary')
plt.show()

#display labels 

print(y_train[1])

print(set(y_train))

## one hot encoding ##

#encoding labels 
from tensorflow.keras.utils import to_categorical 

y_train_encoded=to_categorical(y_train)
y_test_encoded=to_categorical(y_test)

print('y_train_encoded shape: ',y_train_encoded.shape)
print('y_test_encoded shape: ',y_test_encoded.shape)

print(y_train_encoded[1])

## preprocessing the examples ##

#unrolling n-dim arrays to vectors

import numpy as np

x_train_reshaped=np.reshape(x_train,(60000,784))
x_test_reshaped=np.reshape(x_test,(10000,784))

print("x_train_reshaped shape: ",x_train_reshaped.shape)
print("x_test_reshaped shape: ",x_test_reshaped.shape)

#display pixel values
print(set(x_train_reshaped[1]))

#data normalization 

x_mean=np.mean(x_train_reshaped)
x_std=np.std(x_train_reshaped)
epsilon =1e-10

x_train_norm=(x_train_reshaped - x_mean)/(x_std+epsilon)
x_test_norm=(x_test_reshaped - x_mean)/(x_std+epsilon)


print(set(x_train_norm[0]))

## creating a model ##

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model= Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
     Dense(128,activation='relu'),
      Dense(10,activation='softmax')
    ])

#compiling the model 

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

print(model.summary())


##training the model ##

model.fit(x_train_norm,y_train_encoded,epochs=3)

#evaluating the model 

_,accuracy=model.evaluate(x_test_norm,y_test_encoded)
print('test set accuracy: ',accuracy*100)

##predictions##

preds=model.predict(x_test_norm)
print('shape of preds: ',preds.shape)

plt.figure(figsize=(12,12))

start_index=0

for i in range (25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred=np.argmax(preds[start_index+i])
    gt=y_test[start_index+i]
    
    col='g'
    if pred != gt:
        col='r'
    
    plt.xlabel('i={},pred={},gt={}' .format(start_index+i,pred,gt),color=col)
    plt.imshow(x_test[start_index+i],cmap='binary')
        
plt.show()








