import tensorflow as tf;
import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt

dfX = pd.read_csv("C:/dev/data/underFitX.csv");

dfY = pd.read_csv("C:/dev/data/underFitY.csv");

arrX = np.array(dfX)

dfx = np.array(dfX)

for i in range(2, 9) :
    newX = np.pow(dfx, i)
    
    arrX = np.append(arrX, newX, 1)

print(arrX)

dfArrX = pd.DataFrame(arrX);

print(dfArrX);


testDfX = pd.read_csv("C:/dev/data/underFitXTest.csv");

testDfY = pd.read_csv("C:/dev/data/underFitYTest.csv");

normalizer = tf.keras.layers.Normalization()

normalizer.adapt(np.array(dfX));

linearModel = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(1)
])

linearModel.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError, metrics=["accuracy"])

epochsData = linearModel.fit(dfX, dfY, epochs=11, validation_split=0.3);

print(epochsData.history.keys());

print(epochsData.history.values())

print(epochsData.params)

fig, axes = plt.subplots();

x = np.linspace(1, 11, 11)

axes.plot(x, epochsData.history['loss'], label="test")

axes.plot(x, epochsData.history['val_loss'])

# plt.show();

normalizer = tf.keras.layers.Normalization(axis = 1)

normalizer.adapt(arrX);



linearModel2 = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(1)
])

linearModel2.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError, metrics=["accuracy"])

print(arrX.shape)

epochsData = linearModel2.fit(arrX, dfY, epochs=11, validation_split=0.3);

linearModel2.evaluate(testDfX, testDfY)