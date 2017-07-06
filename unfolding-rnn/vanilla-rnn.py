import tensorflow as tf
import numpy as np

xs = tf.placeholder(shape=[None, None], dtype=tf.int32)
ys = tf.placeholder(shape=[None], dtype=tf.int32)

init_state = tf.placeholder(shape=[None, state_size],
                            dtype=tf.float32, name='initial_state')

embs = tf.get_variable('emb', [num_classes, state_size])
rnn_inputs = tf.nn.embedding_lookup(embs, xs)

states = tf.scan(step,
                 tf.transpose(rnn_inputs, [1, 0, 2]),
                 initializer=init_state)

xav_init = tf.contrib.layers.xavier_initializer

W = tf.get_variable('W', shape=[state_size, state_size],
                    initializer=xav_init())
U = tf.get_variable('U', shape=[state_size, state_size],
                    initializer=xav_init())
b = tf.get_variable('b', shape=[state_size],
                    initializer=tf.constant_initializer(0.))

def step(hprev, x):
    return tf.tanh(
        tf.matmul(hprev, W) + 
        tf.matmul(x, U) + b)

V = tf.get_variable('V', shape=[state_size, num_classes],
                   initializer=xav_init())
bo = tf.get_variable('bo', shape=[num_classes],
                     initializer=tf.constant_initializer(0.))
states_reshaped = tf.reshape(states, [-1, state_size])
logits = tf.matmul(states_reshaped, V) + bo
predictions = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys)
loss = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_loss = 0
    for index in range(epochs):
        for batch in range(100):
            xs_, ys_ = train_set.__next__()
            _, train_loss_ = sess.run([train_op, loss],
                                     {xs_ : xs,
                                      ys_ : ys.reshape([batch_size * seqlen]),
                                      init_state : np.zeros([batch_size, state_size])
                                     })
            train_loss += train_loss_
        print('[{}] loss : {}'.format(index, train_loss / 100))
        train_loss = 0
