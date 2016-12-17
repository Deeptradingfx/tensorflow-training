import collections
import tensorflow as tf

session = tf.Session()

a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
v = session.run(a)
print(v)
v = session.run([a, b])
print(v)
MyData = collections.namedtuple('MyData', ['a', 'b'])
v = session.run({'k1' : MyData(a, b), 'k2' : [b, a]})
print(v)
