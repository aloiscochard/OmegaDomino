import tensorflow as tf

execfile('lib/__init__.py')

# TODO More params: learning, dropout, ...

initializer = tf.glorot_uniform_initializer()

output_activation = tf.nn.softmax
hidden_activation = tf.nn.relu

learning_rate = tf.placeholder(tf.float32, [], name="nnet_learning_rate")

inputs = 784
outputs = 10

# layers = [500, 300]
layers = [1200, 600, 300]

nnet_input = tf.placeholder(tf.float32, [None, inputs], name="nnet_input")
nnet_target = tf.placeholder(tf.int32, [None], name="nnet_target")
nnet_target_logits = tf.one_hot(nnet_target, outputs)

nnet_ws = []
nnet_bs = []
nnet_os = []

o_s = inputs
o_p = tf.reshape(nnet_input, [-1, 784])

for i in range(len(layers)):
    if i != 0:
        o_s = layers[i-1]
        o_p = nnet_os[i - 1]

    w = tf.get_variable("nnet_ws_%s"%(i), shape=[o_s, layers[i]], initializer=initializer)
    b = tf.get_variable("nnet_bs_%s"%(i), shape=[layers[i]], initializer=initializer)

    o = hidden_activation(tf.matmul(o_p, w) + b)

    nnet_ws.append(w)
    nnet_bs.append(b)
    nnet_os.append(o)


i = len(layers)
w = tf.get_variable("nnet_ws_%s"%(i), shape=[layers[-1], outputs], initializer=initializer)
b = tf.get_variable("nnet_bs_%s"%(i), shape=[outputs], initializer=initializer)

nnet_ws.append(w)
nnet_bs.append(b)

logits = tf.matmul(nnet_os[-1], w) + b

nnet_output = tf.argmax(output_activation(logits), 1, output_type=tf.int32, name="nnet_output")

if mode == tf.estimator.ModeKeys.PREDICT:
    pass

elif mode == tf.estimator.ModeKeys.EVAL:
    is_correct = tf.equal(nnet_output, nnet_target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="nnet_accuracy")

elif mode == tf.estimator.ModeKeys.TRAIN:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(nnet_target_logits * tf.log(logits), reduction_indices=[1]), name="nnet_cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cross_entropy, name="train")

init = tf.variables_initializer(tf.global_variables(), name='init')

ops = []
for i in range(len(layers) + 1):
    ops.append(nnet_ws[i].assign(tf.placeholder(tf.float32, [None, None], name="nnet_ws_%s_init"%(i))))
    ops.append(nnet_bs[i].assign(tf.placeholder(tf.float32, [None], name="nnet_bs_%s_init"%(i))))

tf.group(ops, name="nnet_init")

network_dump()
