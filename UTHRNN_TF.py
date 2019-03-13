class UTHRNNCell(object):
    def __init__(self,nodes,name):
        self.nodes=nodes
        self.name=name

    def build(self,input_shape):
        self.input_shape=input_shape
        self.input_dim = self.input_shape[-1]
        with tf.variable_scope(self.name):
            self.Wk = tf.get_variable('Wk', shape=[self.input_dim, self.nodes])
            tf.summary.histogram('Wk',self.Wk)
            self.Wr = tf.get_variable('Wr', shape=[self.nodes, self.nodes])
            tf.summary.histogram('Wr', self.Wr)
            self.Ir = tf.get_variable('Initial_state',shape=[self.input_dim,self.nodes],trainable=False)
        print("RNN built: input shape ", self.input_shape)

    def rnn_step(self,previous_output,step_input):
        ci_out=tf.matmul(step_input,self.Wk,name='kernel_mult')
        po_out=tf.matmul(previous_output,self.Wr,name='recurrent_mult')
        step_output=tf.tanh(tf.add(ci_out,po_out))
        return step_output

    def loop_over_timestep(self,input):
        input_tm=tf.transpose(input,[1,0,2])
        initial_state=tf.matmul(input_tm[0],self.Ir)
        output_tm=tf.scan(self.rnn_step,input_tm,initializer=initial_state)
        output=tf.transpose(output_tm,[1,0,2])
        return output
        
def UTHRNN(nodes,input,name,return_time_major=False,return_sequence=True):
    rnncell=psmRNNCell(nodes,name)
    input_shape=get_layer_shape(input)
    rnncell.build(input_shape)
    output=rnncell.loop_over_timestep(input)
    if(not return_sequence):
        output = tf.transpose(output, [1, 0, 2])
        output = output[-1]
        return output
    if(return_time_major):
        output=tf.transpose(output,[1,0,2])
    return output
    
    
def FullyConnected(input,nbnodes,layername,give_prob=False):
    shape=get_layer_shape(input)
    in_dim=shape[-1]
    # print("In dimesion ",in_dim)
    dense_prob=None
    W=tf.Variable(tf.truncated_normal([in_dim,nbnodes]),name=layername+"_W")
    B=tf.constant(0.1,shape=[nbnodes],name=layername+"_B")
    dense_out=tf.matmul(input,W)+B
    if(give_prob):
        dense_prob=tf.nn.softmax(dense_out)
    return dense_out,dense_prob


def get_layer_shape(layer):
	thisshape = tf.Tensor.get_shape(layer)
	ts = [thisshape[i].value for i in range(len(thisshape))]
	return ts
