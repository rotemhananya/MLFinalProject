import numpy as np
import tensorflow as tf

np.random.seed(73)

img = np.random.uniform(0, 255, size=(1, 224, 224, 3))
kernel_in = np.random.uniform(0, 1, size=(3, 3, 1, 1))

np.save('Data/Base Images/Uniform Random')
x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
img = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
print(np.array(img).shape)