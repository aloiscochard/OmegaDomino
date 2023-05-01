import tensorflow as tf

exec(open('lib/__init__.py').read())

with tf.variable_scope("concat"):
    xs = tf.compat.v1.placeholder(tf.float32, [None, None], name="xs")
    ys = tf.compat.v1.placeholder(tf.float32, [None, None], name="ys")
    zs = tf.concat([xs, ys], 0, name="zs")

with tf.variable_scope("gather"):
    xs = tf.compat.v1.placeholder(tf.float32, [None, None], name="xs")
    indices = tf.compat.v1.placeholder(tf.int32, [None], name="indices")
    zs = tf.gather(xs, indices, 0, name="zs")

with tf.variable_scope("reduce_max"):
    xs = tf.compat.v1.placeholder(tf.float32, [None, None], name="xs")
    ys = tf.reduce_max(xs, axis=1, name="zs")

with tf.variable_scope("row_set"):
    xs = tf.placeholder(tf.float32, [None, None], name="xs")

    i  = tf.placeholder(tf.int32, [], name="i")
    x = tf.placeholder(tf.float32, [None, None], name="x")

    bs_idxs = tf.range(0, i, dtype=tf.int32)
    bs = tf.gather(xs, bs_idxs, 0)

    cs_idxs = tf.range(i + 1, tf.shape(xs)[0], dtype=tf.int32)
    cs = tf.gather(xs, cs_idxs, 0)

    zs = tf.concat([bs, x, cs], 0, name="zs")

graph_dump()

