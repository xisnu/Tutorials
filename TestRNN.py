from __future__ import print_function
from UTHRNN_TF import *
import numpy as np


ts=512
f=1
Nc=2

g=tf.Graph()
with g.as_default():
    inp=tf.placeholder(tf.float32,shape=[None,ts,f])
    target=tf.placeholder(tf.float32,shape=[None,Nc])
    rnn1=psmLSTM(4,inp,"rnn1")
    print(get_layer_shape(rnn1))
    rnn2=psmLSTM(8,rnn1,"rnn2",return_sequence=False)
    print(get_layer_shape(rnn2))
    logit,prob=FullyConnected(rnn2,Nc,'output',give_prob=True)
    print(get_layer_shape(logit))
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,logits=logit))
    tf.summary.scalar('loss',loss)
    optimizer=tf.train.RMSPropOptimizer(0.001).minimize(loss)
    actual=tf.argmax(target,axis=-1)
    predicted=tf.argmax(prob,axis=-1)
    corrects=tf.cast(tf.equal(actual,predicted),tf.float32)
    accuracy=tf.reduce_mean(corrects)
    saver=tf.train.Saver()

batchsize=8
epochs=500

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer=tf.summary.FileWriter('./logs/temp3/train',sess.graph)
    # saver.restore(sess,"Weights/psmRNN_eq")
    print("All weights ready")
    x_train,y_train,sl_train=load_timeseries_data(earthquake_path+"TRAIN",Nc,class_label_from_0=True)
    x_train=np.expand_dims(x_train,-1)
    nbtrain=len(x_train)
    batches=int(np.ceil(nbtrain/float(batchsize)))
    tbcounter=0
    merge=tf.summary.merge_all()
    for e in range(epochs):
        start=0
        l=0
        a=0
        for b in range(batches):
            end=min(start+batchsize,nbtrain-1)
            batch_x=x_train[start:end]
            batch_y=y_train[start:end]
            feed={inp:batch_x,target:batch_y}
            bl,_,ba,mrg=sess.run([loss,optimizer,accuracy,merge],feed_dict=feed)
            l+=bl
            a+=ba
            start=end
            summary_writer.add_summary(mrg,tbcounter)
        l=l/float(batches)
        a=a/float(batches)
        saver.save(sess,"Weights/psmRNN_eq")
        print("Loss %f acc %f"%(l,a))
