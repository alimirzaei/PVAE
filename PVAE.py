from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class PVAE():
    def __init__(self, latent_dim = 2, input_dim = 28*28):
        self._latent_dim = latent_dim
        self._input_dim = input_dim
        self.input = Input((input_dim,))
        [mu, log_sigma] = self._encoder(self.input)
        self.z = self._sample(mu, log_sigma)
        self.output = self._decoder(self.z) 
        return None       
        
        
    def getModel(self):
        return Model(self.input, self.output)
    
    def _encoder(self, _input):
        self._batch_size = K.shape(_input)[0]
        self._en1 = Dense(10, activation='relu', name='en1')(_input)
        self._mu = Dense(self._latent_dim, name = 'mu')(self._en1)
        self._log_sigma = Dense(self._latent_dim, name='sigma')(self._en1)
        return [self._mu, self._log_sigma]

    def _sample(self, mu, log_sigma):
        epsilon = K.random_normal(shape=(self._batch_size, self._latent_dim))
        return epsilon * K.exp(log_sigma) + mu

    def _decoder(self, _z):
        self._de1 = Dense(10, activation='relu', name = 'de1')(_z)
        self._output = Dense(self._input_dim, name='output')(self._de1)
        return self._output
    

        


if __name__ == '__main__':
    pvae = PVAE().getModel()
    pvae.summary()
