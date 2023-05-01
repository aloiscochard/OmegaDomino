import ast
import tensorflow as tf

exec(open('lib/__init__.py').read())

initializer = tf.glorot_uniform_initializer()

learning_rate = tf.placeholder(tf.float32, [], name="nnet_learning_rate")
dropout_keep_p = tf.constant(1.0, shape=[]) - tf.placeholder(tf.float32, [], name="nnet_dropout_rate")

output_activation = tf.nn.sigmoid
hidden_activation = tf.nn.relu

inputs = int(sys.argv[2])
outputs = 1

layers = ast.literal_eval(sys.argv[3])

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

    if mode == tf.estimator.ModeKeys.TRAIN:
        o = tf.nn.dropout(o, dropout_keep_p)

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
loss = tf.losses.mean_squared_error(labels=nnet_target_logits, predictions=nnet_output)

if mode == tf.estimator.ModeKeys.PREDICT:
    pass

elif mode == tf.estimator.ModeKeys.EVAL:
    cost = tf.reduce_mean(loss, name="nnet_accuracy")

elif mode == tf.estimator.ModeKeys.TRAIN:
    cost = tf.reduce_mean(loss, name="nnet_cost")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name="train")

init = tf.variables_initializer(tf.global_variables(), name='init')

ops = []
for i in range(len(layers) + 1):
    ops.append(nnet_ws[i].assign(tf.placeholder(tf.float32, [None, None], name="nnet_ws_%s_init"%(i))))
    ops.append(nnet_bs[i].assign(tf.placeholder(tf.float32, [None], name="nnet_bs_%s_init"%(i))))

tf.group(ops, name="nnet_init")

network_dump('snet-%s-%s'%(inputs, layers))
