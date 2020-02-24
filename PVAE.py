from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from keras import objectives
from keras.datasets import mnist
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

class PVAE():
    def __init__(self, latent_dim = 2, input_dim = 28*28):
        self._latent_dim = latent_dim
        self._input_dim = input_dim
        self.input = Input((input_dim,))
        [mu, log_sigma] = self._encoder(self.input)
        self.z = Lambda(lambda x: self._sample(x[0],x[1]))([mu, log_sigma])
        self.output = self._decoder(self.z) 
        self.model = Model(self.input, self.output)
        self.model.compile(optimizer='adam', loss=self.vae_loss)
        return None       
    

        
    def train(self, x_train , epochs=10):
        self.model.fit(x_train, x_train, epochs=epochs)

    def getEncoder(self):
        [mu, log_sigma] = self._encoder(self.input)
        model = Model(self.input, [mu, log_sigma])
        return model

    def getDecoder(self):
        self._decoder_input = Input((self._latent_dim, ))
        self._decoder_output = self._decoder(self._decoder_input)
        return Model(self._decoder_input, self._decoder_output)
    
    def _encoder(self, _input):
        self._batch_size = K.shape(_input)[0]
        self._en1 = Dense(512, activation='relu', name='en1')(_input)
        #self._en2 = Dense(10, activation='relu', name='en2')(self._en1)
        self._mu = Dense(self._latent_dim, name = 'mu')(self._en1)
        self._log_sigma = Dense(self._latent_dim, name='sigma')(self._en1)
        return [self._mu, self._log_sigma]

    def _sample(self, mu, log_sigma):
        epsilon = K.random_normal(shape=(self._batch_size, self._latent_dim))
        return epsilon * K.exp(.5 * log_sigma) + mu

    def _decoder(self, _z):
        self._de1 = Dense(512, activation='relu', name = 'de1')(_z)
        #self._de2 = Dense(100, activation='relu', name = 'de2')(self._de1)
        self._output = Dense(self._input_dim, name='output', activation='sigmoid')(self._de1)
        return self._output

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self._input_dim*objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self._log_sigma - K.square(self._mu) - K.exp(self._log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def generateSamples(self):
        fig = plt.figure(figsize=(20, 20))
        range = np.arange(-3, 3, 1)
        for i, x in enumerate(range):
            for j, y in enumerate(range):
                model = self.getDecoder()
                vector = model.predict(np.array([x, y]).reshape(1,-1)) 
                picture = vector.reshape(28, 28)
                ax = fig.add_subplot(len(range), len(range), i + j*len(range)+1)
                ax.imshow(picture)
        return fig

    def plotSamples(self, x, y):
        model = self.getEncoder() 
        latent = np.array(model.predict(x)[0])
        print(latent.shape)
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter(latent[:,0], latent[:,1], c=y)
        return fig
        
if __name__ == '__main__':
    pvae = PVAE()
    #model = pvae.getModel()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train.reshape(-1, 28*28) / 255.
    x_test = x_test.reshape(-1, 28*28) / 255.
    pvae.train(x,epochs=1)
    fig1 = pvae.generateSamples()
    fig1.savefig('fig1.png')
    fig2 = pvae.plotSamples(x_test, y_test)
    fig2.savefig('fig2.png')

