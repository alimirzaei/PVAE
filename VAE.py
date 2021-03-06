from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

class VariationalAutoEncoder:
    """
    A variational auto-encoder class by Daniel Karavolos, based on:
            https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
    Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
    Example usage:
    original_dim = img_rows * img_cols
    layers = [32, 16]
    vae_obj = VariationalAutoEncoder(original_dim, layers)
    vae = vae_obj.compile()
    vae.summary()
    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=10,
            batch_size=32,
            validation_data=[x_test, x_test], verbose=2)
    x_train_encoded = vae_obj.encoder.predict(x_train, batch_size=batch_size)
    x_decoded = vae_obj.decoder.predict(z_sample, batch_size=batch_size)
    """

    def __init__(self, input_size, d_layers, activation='relu', optimizer='adam', show_metrics=False, dropout=0.0, model_address=None):
        # variables
        self.input_size = input_size
        self.layer_sizes = d_layers
        self.latent_dim = 2
        self.drop_prob = float(dropout)
        # building tensors
        self.input = Input(shape=(input_size,), name="encoder_input")
        self.z_mean, self.z_var = self.create_encoder(self.input, activation, self.drop_prob)
        self.output, self.decoder = self.create_decoder(activation, self.drop_prob)
        self.encoder = Model(self.input, self.z_mean)
        self.model = Model(self.input, self.output)
        if model_address:
            self.model.load_weights(model_address)
        self.optimizer = optimizer
        self.verbose = show_metrics

    
    # returns two tensors, one for the encoding (z_mean), one for making the manifold smooth
    def create_encoder(self, nn_input, act, drop):
        x = nn_input
        for l in self.layer_sizes:
            x = Dense(l, activation=act)(x)
            if drop:
                x = Dropout(drop)(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        return z_mean, z_log_var

    # returns the output tensor of the auto-encoder and de decoder model.
    def create_decoder(self, act, drop):
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_var])
        rev_layers = self.layer_sizes[::-1]
        # Build sampler (for training) and decoder at the same time.
        ae_output = z
        inpt = Input(shape=(self.latent_dim,), name="decoder_input")
        dec_tensor = inpt
        if len(rev_layers) > 1:
            for lay in rev_layers:
                dec = Dense(lay, activation=act)
                ae_output = dec(ae_output)
                dec_tensor = dec(dec_tensor)
                if drop:
                    ae_output = Dropout(drop)(ae_output)
                    dec_tensor = Dropout(drop)(dec_tensor)
        output_layer = Dense(self.input_size, activation=act, name="output")
        ae_output = output_layer(ae_output)
        dec_tensor = output_layer(dec_tensor)
        # dec_tensor is turned into a model. ae_output will be turned into a Model in the __init__
        decoder = Model(inpt, dec_tensor)
        return ae_output, decoder

    # used for training
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2.) * epsilon

    # loss functions
    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.reconstruction_loss(x, x_decoded_mean)
        kl_loss = self.kl_loss(x, x_decoded_mean)
        return K.mean(xent_loss + kl_loss)

    def reconstruction_loss(self, x, x_decoded_mean):
        return self.input_size * metrics.mse(x, x_decoded_mean)

    def kl_loss(self, x, x_decoded_mean):   # inputs are here so you can use it as a metric
        return - 0.5 * K.sum(1 + self.z_var - K.square(self.z_mean) - K.exp(self.z_var), axis=-1)

    # builds and returns the model. This is how you get the model in your training code.
    def compile(self, modelAddress = None):
        met = []
        if self.verbose:
            met = [self.reconstruction_loss, self.kl_loss]
        self.model.compile(self.optimizer, loss=self.vae_loss, metrics=met)
        if(modelAddress):
            self.load(modelAddress)
        return self.model
    
    def save(self, modelAddres = 'vae.h5'):
        self.model.save_weights(modelAddres)
    
    def load(self, modelAddres = 'vae.h5'):
        self.model.load_weights(modelAddres)
    
    def generateSamples(self):
        fig = plt.figure(figsize=(20, 20))
        range_latent = np.arange(-3, 3, 1)
        N = len(range_latent)
        for i, x in enumerate(range_latent):
            for j, y in enumerate(range_latent):
                vector = self.decoder.predict(np.array([x, y]).reshape(1,-1)) 
                picture = vector.reshape(28, 28)
                ax = fig.add_subplot(N, N, i + j*N+1)
                ax.imshow(picture)
        return fig

    def plotSamples(self, x, y):
        latent = np.array(self.encoder.predict(x))
        print(latent.shape)
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter(latent[:,0], latent[:,1], c=y)
        return fig



if __name__ == '__main__':
    vae_obj = VariationalAutoEncoder(28*28, [32, 16])
    vae = vae_obj.compile()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train.reshape(-1, 28*28) / 256.
    x_test = x_test.reshape(-1, 28*28) / 256.
    vae.fit(x,x,epochs=5)
    fig1 = vae_obj.generateSamples()
    fig1.savefig('fig1.png')
    fig2 = vae_obj.plotSamples(x_test, y_test)
    fig2.savefig('fig2.png')
