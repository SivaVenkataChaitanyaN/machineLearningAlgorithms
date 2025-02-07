import tensorflow as tf;
import pandas as pd;
import numpy as np;

columns = ['X', 'Y'];

df = pd.read_csv('C:/dev/data/twoVariableData.csv', names=columns)

trainDf = df.sample(frac=0.8)

testDf = df.drop(trainDf.index)

trainY = trainDf.pop('Y')

testY = testDf.pop('Y')

print(trainDf)

print(trainY)

normalizer = tf.keras.layers.Normalization()

normalizer.adapt(np.array(trainDf))

linearModel = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(1)
 ])

linearModel.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError, metrics=["accuracy"])

linearModel.fit(trainDf, trainY, epochs=80)

linearModel.evaluate(testDf, testY)

predictDf = pd.DataFrame([12, 5.7077]);

print(predictDf);

predicted = linearModel.predict(np.array(predictDf))

print(predicted)