import tensorflow as tf;
import pandas as pd;
import tensorflow_probability as tfp;
import numpy as np;

df = pd.read_csv("C:/dev/data/logisticData.csv");

Y = tf.reshape(tf.constant(df.iloc[:, 2], dtype='float64'), [99, 1]);

onesDf = pd.DataFrame(np.ones(len(df)));

X = onesDf.join(df.iloc[:, 0:2]);

X = tf.constant(X, dtype='float64');

A = tf.Variable(pd.DataFrame([0, 0, 0]), dtype='float64');

one = tf.constant(1, dtype='float64');

@tf.function
def costFunction():
    return tf.math.reduce_mean(tf.math.multiply(-Y, tf.math.log(tf.math.sigmoid(X @ A))) - tf.math.multiply(tf.math.subtract(one, Y), tf.math.log(tf.math.subtract(one, tf.math.sigmoid(X @ A)))), 0);
    

print(costFunction())

adam = tf.keras.optimizers.Adam(learning_rate=0.001);

print(tfp.math.minimize(costFunction, num_steps=2000, optimizer=adam));
    
print(A);


