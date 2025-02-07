import tensorflow as tf;
import pandas as pd;

model = tf.keras.Sequential()

model.add(tf.keras.Input(shape=(400,)))

model.add(tf.keras.layers.Dense(100, activation="sigmoid"))

model.add(tf.keras.layers.Dense(25, activation="sigmoid"))

model.add(tf.keras.layers.Dense(10, activation="sigmoid"));

print(model.weights);

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

print(model.to_json())

dfx = pd.read_csv("C:/dev/data/dataX.csv");

dfy = pd.read_csv("C:/dev/data/dataY.csv");

X = tf.constant(dfx, dtype='float64', shape=[4999, 400]);

Y = tf.constant(dfy, dtype='float64', shape=[4999, 10]);

model.fit(X, Y, epochs=150);

print(model.weights);

print(model.export("model_weights.txt"));

