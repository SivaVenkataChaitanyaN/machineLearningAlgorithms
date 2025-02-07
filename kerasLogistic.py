import tensorflow as tf;
import pandas as pd;
import numpy as np;
import datetime;

columns = ['X0', 'X1', 'Y']

df = pd.read_csv("C:/dev/data/logisticData.csv", names=columns);

trainDf = df.sample(frac=0.8);

testDf = df.drop(trainDf.index);

trainY = trainDf.pop('Y');

testY = testDf.pop('Y');

print(trainDf);

normalizer = tf.keras.layers.Normalization();

normalizer.adapt(np.array(trainDf))

print(normalizer.mean.numpy())

logisticModel = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(1)
]);

logsDir = "C:/dev/code/python/machineLearning/tensorBoard/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S");

tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=logsDir, histogram_freq=1)

logisticModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

logisticModel.fit(trainDf, trainY, epochs=100, callbacks=[tensorboardCallback])

score = logisticModel.evaluate(testDf, testY, return_dict=True);

print(score);
# print("Loss: ",score[0]);

# print("Accuracy: ", score[1]);