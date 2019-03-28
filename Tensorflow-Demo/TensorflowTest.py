from __future__ import print_function
import tensorflow as tf
import numpy as np

yourgraph=tf.Graph()

with yourgraph.as_default():
    x = tf.placeholder(tf.float32,shape=[None])
    y = tf.placeholder(tf.float32, shape=[None])
    node_plus = tf.add(x,y)
    node_multiply = tf.multiply(x,y)
    node_minus = tf.subtract(node_plus,node_multiply)
    print("Graph Ready")

x_data = np.random.randint(0,high=10,size=[3])
y_data = np.random.randint(0,high=10,size=[3])
print(x_data,y_data)


with tf.Session(graph=yourgraph) as s:
    feed={x:x_data,y:y_data}
    out=s.run([node_minus],feed_dict=feed)
    out=np.asarray(out)
    print(out)
