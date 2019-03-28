from __future__ import print_function

import numpy as np
import tensorflow as tf
from ReadData import *

x_data, y_data = load_soybean_data("Data/soybean-small.data")

nb_features=x_data.shape[1]
nb_classes=y_data.shape[1]

print("Number of Features ",nb_features," Number of classes ",nb_classes)

mlpgraph=tf.Graph()
with mlpgraph.as_default():
    x=tf.placeholder(tf.float32,[None,nb_features])
    y=tf.placeholder(tf.float32,[None,nb_classes])

    w12=tf.Variable(tf.truncated_normal([nb_features,16],stddev=0.1),name="W12")
    b1=tf.constant(0.1,shape=[16])
    input_h1=tf.matmul(x,w12)
    input_h1_nl=tf.tanh(input_h1+b1)

    w22 = tf.Variable(tf.truncated_normal([16,24], stddev=0.1), name="W22")
    b2 = tf.constant(0.1, shape=[24])
    h1_h2 = tf.matmul(input_h1_nl, w22)
    h1_h2_nl = tf.tanh(h1_h2+b2)

    w23=tf.Variable(tf.truncated_normal([24,nb_classes], stddev=0.1), name="w23")
    b3 = tf.constant(0.1, shape=[nb_classes])
    h2_out=tf.matmul(h1_h2_nl,w23) + b3

    h2_out_softmax=tf.nn.softmax(h2_out)

    loss=tf.nn.softmax_cross_entropy_with_logits(logits=h2_out,labels=y)
    loss=tf.reduce_mean(loss)
    optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    true=tf.argmax(y,axis=1)
    pred=tf.argmax(h2_out,axis=1)
    corrects=tf.cast(tf.equal(true,pred),tf.float32)
    acc=tf.reduce_mean(corrects)

    print("Graph Ready")

with tf.Session(graph=mlpgraph) as yoursession:
    init=tf.global_variables_initializer()
    yoursession.run([init])
    print("Weights are initialized")

    total_samples=x_data.shape[0]
    batch_size=8
    epochs=1500
    nb_batches = int(np.ceil(total_samples /float(batch_size)))
    for e in range(epochs):
        total_loss=0
        total_acc=0
        start=0
        for b in range(nb_batches):
            end=min(total_samples-1,start+batch_size)
            #print("\tstart ", start," end ",end)
            x_train=x_data[start:end]
            y_train=y_data[start:end]
            feed={x:x_train,y:y_train}
            _,bl,ba=yoursession.run([optimizer,loss,acc],feed_dict=feed)
            total_loss+=bl
            total_acc+=ba
            start=end

        total_loss=total_loss/nb_batches
        total_acc=total_acc/nb_batches
        print("Epoch %d loss %0.4f acc %0.3f"%(e,total_loss,total_acc))