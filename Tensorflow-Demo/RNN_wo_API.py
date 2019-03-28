import numpy as np
import tensorflow as tf


def get_layer_shape(layer):
    thisshape = tf.Tensor.get_shape(layer)
    ts = [thisshape[i].value for i in range(len(thisshape))]
    return ts


class psmRNNCell_v2(object):
    def __init__(self, nodes, name):
        self.nodes = nodes
        self.name = name

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.T = input_shape[0]  # time major
        self.W = tf.get_variable('W_' + self.name, shape=[self.input_dim, self.nodes],
                                 dtype=tf.float32)  # F,H ,initializer='random_uniform'
        tf.summary.histogram('W_' + self.name, self.W)
        self.U = tf.get_variable('U_' + self.name, shape=[self.nodes, self.nodes],
                                 dtype=tf.float32)  # H,H,initializer='random_uniform'
        tf.summary.histogram('U_' + self.name, self.U)
        self.bias = tf.get_variable('Bias_' + self.name, shape=[self.nodes],
                                    dtype=tf.float32)  # H,initializer='random_uniform'
        self.Winit = tf.zeros([self.input_dim, self.nodes], dtype=tf.float32, name='Winit')

    def stop_loop_if(self, t, prev_out, output_sequence):
        return tf.less(t, self.T)

    def rnn_op(self, t, prev_out, output_sequence):
        current_input = self.input_sequence[t]  # N,F
        current_output = tf.tanh(tf.matmul(current_input, self.W, name='input_nodes') + tf.matmul(prev_out, self.U,
                                                                                                  name='nodes_nodes') + self.bias)  # N,H
        output_sequence = output_sequence.write(t, current_output)
        t = t + 1
        return [t, current_output, output_sequence]

    def rnn(self, input_sequence):
        self.input_sequence = tf.transpose(input_sequence, [1, 0, 2])  # convert to time major
        initial_state = tf.matmul(self.input_sequence[0], self.Winit, name='initial_state')  # N,H
        output_sequence = tf.TensorArray(tf.float32, size=self.T)
        T, _, Y = tf.while_loop(self.stop_loop_if, self.rnn_op, [0, initial_state, output_sequence])
        Y = Y.concat()
        Y = tf.reshape(Y, [self.T, -1, self.nodes])
        return Y


def psmRNN_v2(nodes, input, name, return_time_major=False, return_sequence=True):
    rnncell = psmRNNCell_v2(nodes, name)
    shape = get_layer_shape(input)
    rnncell.build([shape[1], shape[0], shape[2]])
    output = rnncell.rnn(input)
    if (not return_sequence):
        return output[-1]
    if (not return_time_major):
        output = tf.transpose(output, [1, 0, 2])
    return output


def FullyConnected(input, nbnodes, layername, give_prob=False):
    shape = get_layer_shape(input)
    in_dim = shape[-1]
    # print("In dimesion ",in_dim)
    dense_prob = None
    W = tf.Variable(tf.truncated_normal([in_dim, nbnodes]), name=layername + "_W")
    B = tf.constant(0.1, shape=[nbnodes], name=layername + "_B")
    dense_out = tf.matmul(input, W) + B
    if (give_prob):
        dense_prob = tf.nn.softmax(dense_out)
    return dense_out, dense_prob


def load_timeseries_data(timeseries_csv, nbclass, class_label_from_0=False):
    f = open(timeseries_csv)
    line = f.readline()
    sequence_length = []
    labels = []
    features = []
    while line:
        info = line.strip("\n").split(",")
        label = info[0]
        label_one_hot = np.zeros([nbclass])
        if (class_label_from_0):
            label_index = int(label)
        else:
            label_index = int(label) - 1
        label_one_hot[label_index] = 1
        feat = info[1:]
        nbfeatures = len(feat)
        sequence_length.append(nbfeatures)
        labels.append(label_one_hot)
        features.append(feat)
        line = f.readline()
    max_length = max(sequence_length)
    min_length = min(sequence_length)
    print("Maximum sequence length: ", max_length, " minimum sequence length: ", min_length)
    return np.asarray(features), np.asarray(labels), sequence_length


ts = 512
f = 1
Nc = 2

g = tf.Graph()
with g.as_default():
    inp = tf.placeholder(tf.float32, shape=[None, ts, f])
    target = tf.placeholder(tf.float32, shape=[None, Nc])
    rnn1 = psmRNN_v2(4, inp, "rnn1")
    print("RNN 1=", get_layer_shape(rnn1))
    rnn2 = psmRNN_v2(8, rnn1, "rnn2", return_sequence=False)
    print("RNN 2=", get_layer_shape(rnn2))
    logit, prob = FullyConnected(rnn2, Nc, 'output', give_prob=True)
    print(get_layer_shape(logit))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=logit))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss)
    actual = tf.argmax(target, axis=-1)
    predicted = tf.argmax(prob, axis=-1)
    corrects = tf.cast(tf.equal(actual, predicted), tf.float32)
    accuracy = tf.reduce_mean(corrects)
    saver = tf.train.Saver()

batchsize = 8
epochs = 500
earthquake_path = "Data/Earthquakes_"

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('./logs/temp3/train', sess.graph)
    # saver.restore(sess,"Weights/psmRNN_eq")
    print("All weights ready")
    x_train, y_train, sl_train = load_timeseries_data(earthquake_path + "TRAIN", Nc, class_label_from_0=True)
    x_train = np.expand_dims(x_train, -1)
    nbtrain = len(x_train)
    batches = int(np.ceil(nbtrain / float(batchsize)))
    tbcounter = 0
    merge = tf.summary.merge_all()
    for e in range(epochs):
        start = 0
        l = 0
        a = 0
        for b in range(batches):
            end = min(start + batchsize, nbtrain - 1)
            batch_x = x_train[start:end]
            batch_y = y_train[start:end]
            feed = {inp: batch_x, target: batch_y}
            bl, _, ba, mrg = sess.run([loss, optimizer, accuracy, merge], feed_dict=feed)
            l += bl
            a += ba
            start = end
            summary_writer.add_summary(mrg, tbcounter)
        l = l / float(batches)
        a = a / float(batches)
        saver.save(sess, "Weights/psmRNN_eq")
        print("Loss %f acc %f" % (l, a))
