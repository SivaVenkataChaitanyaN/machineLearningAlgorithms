import tensorflow as tf;
import numpy as np;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
import pandas as pd;

dfX = pd.read_csv("C:/dev/data/ex8/MultiVariantX.csv", header=None);

npX = dfX.to_numpy();

figure, axes = plt.subplots();

axes.plot(npX[:, 0], npX[:, 1], 'bx');

gm = tfp.distributions.MixtureSameFamily(
  mixture_distribution=tfp.distributions.Categorical(
	  probs=[0.2, 0.4, 0.4]),
  components_distribution=tfp.distributions.MultivariateNormalDiag(
	  loc=[[-1., 1],  
		   [1, -1],  
		   [1, 1]],  
	  scale_diag=tf.tile([[.3], [.6], [.7]], [1, 2])))

x1, x2 = np.meshgrid(np.arange(0, 36, 0.5), np.arange(0, 36, 0.5), indexing='ij')

plotX = np.array([x1.ravel(), x2.ravel()])

plotX = plotX.T.reshape(72, 72, 2)

axes.contour(x1, x2, gm.prob(plotX));

plt.show();