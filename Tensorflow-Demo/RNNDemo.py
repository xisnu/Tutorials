from __future__ import print_function
import numpy as np
import tensorflow as tf


def get_layer_shape(layer):
    thisshape = tf.Tensor.get_shape(layer)
    ts = [thisshape[i].value for i in range(len(thisshape))]
    return ts

def load_data(filename):
    f=open(filename)
    line=f.readline()
    xs=[]
    ys=[]
    while line:
        items=line.strip("\n").split(",")
        targetname=items[-1]
        target_index=int(targetname[-1])-1
        target_1h=[0,0,0,0]
        target_1h[target_index]=1.0
        features=items[:-1]
        xs.append(features)
        ys.append(target_1h)
        line=f.readline()
    xs=np.expand_dims(xs,2)
    ys=np.asarray(ys)
    print("X Shape ",xs.shape," Y Shape ",ys.shape)
    return xs,ys

x_data,y_data=load_data("Data/soybean-small.data")

nb_features=x_data.shape[1]
nb_classes=y_data.shape[1]

rnngraph=tf.Graph()
with rnngraph.as_default():
    x = tf.placeholder(tf.float32, [None, nb_features, 1])
    y = tf.placeholder(tf.float32, [None, nb_classes])

    cell = tf.nn.rnn_cell.LSTMCell(32, state_is_tuple=True)
    val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    time_major=tf.transpose(val,[1,0,2])
    last_step = time_major[-1]

    W = tf.Variable(tf.truncated_normal([32,nb_classes],stddev=0.1))
    B =  tf.constant(0.1,shape=[nb_classes])
    dense = tf.matmul(last_step,W)+B

    dense_softmax = tf.nn.softmax(dense)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=y)
    mean_loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(mean_loss)

    true = tf.argmax(y, axis=1)
    pred = tf.argmax(dense_softmax, axis=1)
    corrects = tf.cast(tf.equal(true, pred), tf.float32)
    acc = tf.reduce_mean(corrects)

    #shape=get_layer_shape(dense)
    print("RNN Ready")


with tf.Session(graph=rnngraph) as yoursession:
    init=tf.global_variables_initializer()
    yoursession.run([init])
    print("Weights are initialized")

    total_samples=x_data.shape[0]
    batch_size=8
    epochs=1500
    nb_batches = int(np.ceil(total_samples / float(batch_size)))
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
            _,bl,ba=yoursession.run([optimizer,mean_loss,acc],feed_dict=feed)
            total_loss+=bl
            total_acc+=ba
            start=end

        total_loss=total_loss/nb_batches
        total_acc=total_acc/nb_batches
        print("Epoch %d loss %0.4f acc %0.3f"%(e,total_loss,total_acc))