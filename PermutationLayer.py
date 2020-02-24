from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.initializers import glorot_normal

class NaNHandlingLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self._K = output_dim
        super(NaNHandlingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self._obs_dim = int(input_shape[0][1])
        # Create a trainable weight variable for this layer.
        self.F = self.add_weight(name='F',
                                      shape=(1, self._obs_dim, 10),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.b =  self.add_weight(name = "b",
            shape=[1, self._obs_dim, 1],
            initializer=glorot_normal(),
            trainable=True)
        super(NaNHandlingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        observations, mask = x
        self._batch_size = K.shape(observations)[0]
        self.x_flat = K.reshape(observations, [-1, 1])
        F_flat = K.tile(self.F, [self._batch_size, 1, 1])
        F_flat = K.reshape(F_flat, [-1, 10])
        b_flat = K.tile(self.b, [self._batch_size, 1, 1])
        b_flat = K.reshape(b_flat, [-1, 1])
        # self.x_aug = K.concatenate(
        #     [self.x_flat, self.x_flat * F_flat, b_flat], 1)
        self.x_aug = K.concatenate(
            [self.x_flat * F_flat, b_flat], 1)
        print('x_aug', self.x_aug.shape)
        self.encoded = Dense(self._K)(self.x_aug) #layers.fully_connected(self.x_aug, self._K)
        print('e1', self.encoded)
        self.encoded = K.reshape(self.encoded,
                                    [-1, self._obs_dim, self._K])
        print('e2', self.encoded)
        self.mask_on_hidden = K.reshape(mask,
                                            [-1, self._obs_dim, 1])
        print('make', self.mask_on_hidden)
        self.mask_on_hidden = K.tile(self.mask_on_hidden,
                                        [1, 1, self._K])
        print('mask2', self.mask_on_hidden)
        print(self.encoded.shape)
        self.encoded = K.relu(
            K.sum(self.encoded * self.mask_on_hidden, 1))
        return self.encoded

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], self._K)