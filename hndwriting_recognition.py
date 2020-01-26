import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import tensorflow.keras.datasets.mnist as mnist

num_classes = 10
num_features = 784

(xtrain,ytrain), (xtest,ytest) = mnist.load_data()

(xtrain,xtest) = np.array(xtrain,np.float32), np.array(xtest,np.float32)
(xtrain,xtest) = xtrain.reshape([-1,num_features]), xtest.reshape([-1,num_features])
(xtrain,xtest) = xtrain/255, xtest/255


def display_sample (num):
    label = ytrain[num]
    image = xtrain[num].reshape([28,28])
    plt.title("Label: %d" %(label))
    plt.imshow(image)
    plt.show()
    
#display_sample(977)

learning_steps = 0.001
training_steps = 3000
batch_size = 128
display_steps = 100
n_hidden = 512
train_data = tf.data.Dataset.from_tensor_slices((xtrain,ytrain))
train_data = train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)
randomNormal = tf.initializers.RandomNormal()

weights = {
    
    'h' : tf.Variable(randomNormal([num_features,n_hidden])),
    'out' : tf.Variable(randomNormal([n_hidden,num_classes]))
}
bias = {
    
    'h' : tf.Variable(randomNormal([num_features,n_hidden])),
    'out' : tf.Variable(randomNormal([n_hidden,num_classes]))
}