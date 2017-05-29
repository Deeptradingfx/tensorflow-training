import tensorflow as tf

def main():
    with tf.device('/gpu:0'):
        W = tf.Variable([.3], tf.float32)
        b = tf.Variable([-.3], tf.float32)
        x = tf.placeholder(tf.float32)
        linear_model = W * x + b
        y = tf.placeholder(tf.float32)
        squared_deltas = tf.square(linear_model - y)
        loss = tf.reduce_sum(squared_deltas)

        init = tf.global_variables_initializer()

        train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        for i in range(1000):
            sess.run(train, {x: [1,2,3,4], y: [0,-1,-2,-3]})
            curr_w, curr_b, curr_loss = sess.run([W, b, loss], {x: [1,2,3,4], y: [0,-1,-2,-3]})
        print("W: {}, b: {}, loss: {}".format(curr_w, curr_b, curr_loss))
#        print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
#        print(sess.run(loss, {x:[1,2,3,4], y: [0,-1,-2,-3]}))
#        fixW = tf.assign(W, [-1.])
#        fixb = tf.assign(b, [1.])
#        sess.run([fixW, fixb])
#        print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
        

if __name__ == '__main__':
    main()
