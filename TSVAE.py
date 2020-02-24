from VAE import VariationalAutoEncoder
from keras.datasets import mnist
import numpy as np
from scipy.sparse import random
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from PermutationLayer import NaNHandlingLayer
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 256.
x_test = x_test.reshape(-1, 28*28) / 256.


vae_obj = VariationalAutoEncoder(28*28, [32, 16], model_address='vae.h5')

latents = vae_obj.encoder.predict(x_train)

mask = np.vstack([(random(1, x_train.shape[1],density=np.random.rand()).todense()>0).astype(np.float32) for instance in x_train])
# x_train[np.where(mask <0.5)] = np.nan

x_tensor = Input((28*28,))
mask_tensor = Input((28*28,))
rep_tensor = NaNHandlingLayer(500)([x_tensor, mask_tensor])
output_tensor = Dense(2)(rep_tensor)

model = Model([x_tensor, mask_tensor], output_tensor)
model.compile(optimizer='adam', loss='mse')
#model.load_weights('student.h5')
model.fit([x_train, mask], latents, epochs= 100)
model.save_weights('student.h5')
NTEST = 100
x_test = x_test[:NTEST]
mask_test = np.vstack([(random(1, x_test.shape[1],density=np.random.rand()).todense()>0).astype(np.float32) for instance in x_test])
feeded = np.copy(x_test)
feeded[np.where(mask_test == 0)] = 0
codes = model.predict([x_test, mask_test])
reconstructed = vae_obj.decoder.predict(codes)
fig = plt.figure(figsize=(4, NTEST))
for index in range(len(x_test)):
    ax0 = fig.add_subplot(4, NTEST, index+1)
    ax0.imshow(x_test[index].reshape(28, 28))
    ax1 = fig.add_subplot(4, NTEST, NTEST + index+1)
    ax1.imshow(mask_test[index].reshape(28, 28))
    ax2 = fig.add_subplot(4, NTEST, 2*NTEST + index+1)
    ax2.imshow(reconstructed[index].reshape(28, 28))
    ax3 = fig.add_subplot(4, NTEST, 3*NTEST + index+1)
    ax3.imshow(feeded[index].reshape(28, 28))
fig.save_fig('recons.png')
