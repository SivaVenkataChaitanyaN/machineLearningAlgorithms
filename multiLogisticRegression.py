import tensorflow as tf;
import pandas as pd;
import tensorflow_probability as tfp;
import numpy as np;

dfx = pd.read_csv("C:/dev/data/imageDataX.csv");

dfy = pd.read_csv("C:/dev/data/imageDataY.csv");

X = tf.constant(dfx, dtype='float64');

Y = tf.constant(dfy, dtype='float64');

l = tf.constant(0.1/(2*4999), dtype='float64'); 

A = tf.Variable(pd.DataFrame(np.zeros(400)));

one = tf.constant(1, dtype='float64');

print((tf.math.reduce_sum(tf.math.square(A)) - A[0]) * l);

@tf.function
def costFunction():
    return tf.math.reduce_mean(tf.math.multiply(-Y, tf.math.log(tf.math.sigmoid(X @ A))) - tf.math.multiply(tf.math.subtract(one, Y), tf.math.log(tf.math.subtract(one, tf.math.sigmoid(X @ A)))), 0) + ((tf.math.reduce_sum(tf.math.square(A)) - A[0]) * l);


adam = tf.keras.optimizers.Adam(learning_rate=0.001);

for i in range(0, 11):
    print(i);
    dfv = dfy == i;
    dfv.astype(int);
    Y = tf.constant(dfv, dtype='float64');
    print(tfp.math.minimize(costFunction, num_steps=2000, optimizer=adam));
    print(A);


