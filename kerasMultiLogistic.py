import pandas as pd;
import tensorflow as tf;
import random;
import numpy as np;

dfX = pd.read_csv("C:/dev/data/imageDataXKeras.csv", header=None);

dfY = pd.read_csv("C:/dev/data/imageDataYKeras.csv", header=None);

index = set()

while(len(index) != 3000) :
    n = random.randrange(1, 5000)
    index.add(n)


trainDfX = dfX.drop(index)

trainDfX2 = dfX.iloc[:, 1:201]

trainDfX2 = trainDfX2.drop(index)

trainDfY = dfY.drop(index);

testDfX = dfX.drop(trainDfX.index);

testDfX2 =  dfX.iloc[:, 1:201].drop(trainDfX2.index)

testDfY = dfY.drop(trainDfY.index);

inputs = [];

inputs.append(tf.keras.Input(shape=(400,)))

inputs.append(tf.keras.Input(shape=(200,)))

middle = tf.keras.layers.Dense(100, activation="relu")(inputs[0])

middle2 = tf.keras.layers.Dense(25, activation="relu")(middle)

outputs = tf.keras.layers.Dense(10, activation="softmax")(middle2)

model = tf.keras.Model(inputs= inputs, outputs=outputs)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.RMSprop(), metrics=["accuracy"])

printWeights = lambda logs: print(model.layers[0].get_weights);

weightCallBack = tf.keras.callbacks.LambdaCallback(on_epoch_begin=printWeights, on_epoch_end=printWeights, on_train_begin=printWeights, on_train_end=printWeights, on_train_batch_begin=printWeights, on_train_batch_end=printWeights)

model.fit([trainDfX, trainDfX2], trainDfY, batch_size=64, epochs=1, validation_split=0.2, callbacks=weightCallBack)

print(model.layers[0].get_weights)

print(model.layers[1].get_weights)

model.evaluate([testDfX, testDfX2], testDfY);

print(model.summary())

#print(model.predict(np.array(dfX.iloc[1:3, :])))
