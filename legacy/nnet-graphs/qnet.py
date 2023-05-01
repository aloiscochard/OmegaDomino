import ast
import tensorflow as tf

exec(open('lib/__init__.py').read())

initializer = tf.glorot_uniform_initializer()

clipping = 1.0

learning_rate = tf.placeholder(tf.float32, [], name="nnet_learning_rate")
dropout_keep_p = tf.constant(1.0, shape=[]) - tf.placeholder(tf.float32, [], name="nnet_dropout_rate")

tpe = sys.argv[2]

output_activation = tf.nn.softmax if 'p' in tpe else tf.nn.tanh
hidden_activation = tf.nn.leaky_relu

inputs = int(sys.argv[3])
outputs = int(sys.argv[4])

layers = ast.literal_eval(sys.argv[5])

nnet_input = tf.placeholder(tf.float32, [None, inputs], name="nnet_input")
nnet_target_logits = tf.placeholder(tf.float32, [None, outputs], name="nnet_target")

nnet_ws = []
nnet_bs = []
nnet_os = []

o_s = inputs
o_p = tf.reshape(nnet_input, [-1, inputs])

for i in range(len(layers)):
    if i != 0:
        o_s = layers[i-1]
        o_p = nnet_os[i - 1]

    w = tf.get_variable("nnet_ws_%s"%(i), shape=[o_s, layers[i]], initializer=initializer)
    b = tf.get_variable("nnet_bs_%s"%(i), shape=[layers[i]], initializer=initializer)

    o = hidden_activation(tf.matmul(o_p, w) + b)

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     o = tf.nn.dropout(o, dropout_keep_p)

    nnet_ws.append(w)
    nnet_bs.append(b)
    nnet_os.append(o)


i = len(layers)
w = tf.get_variable("nnet_ws_%s"%(i), shape=[layers[-1], outputs], initializer=initializer)
b = tf.get_variable("nnet_bs_%s"%(i), shape=[outputs], initializer=initializer)

nnet_ws.append(w)
nnet_bs.append(b)

logits = tf.matmul(nnet_os[-1], w) + b
nnet_output = output_activation(logits, name="nnet_output")

def cross_entropy():
    return tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(nnet_target_logits, 1), logits=logits)

def quadratic():
    return tf.losses.mean_squared_error(labels=nnet_target_logits, predictions=nnet_output)

if mode == tf.estimator.ModeKeys.PREDICT:
    nnet_output_max = tf.reduce_max(nnet_output, axis=1, name="nnet_output_max")

elif mode == tf.estimator.ModeKeys.EVAL:
    is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(nnet_target_logits, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="nnet_accuracy")

elif mode == tf.estimator.ModeKeys.TRAIN:
    loss = cross_entropy() if "p" in tpe else quadratic()
    cost = tf.reduce_mean(loss, name="nnet_cost")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, clipping)

    gradients_max = tf.reduce_max([tf.reduce_max(g) for g in gradients], name="nnet_gradients_max")
    gradients_min = tf.reduce_min([tf.reduce_min(g) for g in gradients], name="nnet_gradients_min")

    optimize = optimizer.apply_gradients(zip(gradients, variables), name="train")

    optimizer_init = tf.variables_initializer(optimizer.variables(), name="nnet_optimizer_init")

init = tf.variables_initializer(tf.global_variables(), name='init')

ops = []
for i in range(len(layers) + 1):
    ops.append(nnet_ws[i].assign(tf.placeholder(tf.float32, [None, None], name="nnet_ws_%s_init"%(i))))
    ops.append(nnet_bs[i].assign(tf.placeholder(tf.float32, [None], name="nnet_bs_%s_init"%(i))))

tf.group(ops, name="nnet_init")

network_dump('qnet-%s-%s-%s-%s'%(tpe, inputs, layers, outputs))
