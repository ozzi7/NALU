"""
Learns a+b+c+d using 3 NALUs
https://arxiv.org/abs/1808.00508
"""
import numpy as np
import keras.backend as K
from keras.layers import *
from keras.initializers import *
from keras.models import *


class NALU(Layer):
    def __init__(self, units, MW_initializer='glorot_uniform',
                 G_initializer='glorot_uniform', mode="NALU",
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.MW_initializer = initializers.get(MW_initializer)
        self.G_initializer = initializers.get(G_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='M_hat')
        if self.mode == "NALU":
            self.G = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.G_initializer,
                                     name='G')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)
        if self.mode == "NAC":
            output = a
        elif self.mode == "NALU":
            m = K.exp(K.dot(K.log(K.abs(inputs) + 1e-7), W))
            g = K.sigmoid(K.dot(K.abs(inputs), self.G))
            output = g * a + (1 - g) * m
        else:
            raise ValueError("Valid modes: 'NAC', 'NALU'.")
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'mode': self.mode,
            'MW_initializer': initializers.serialize(self.MW_initializer),
            'G_initializer': initializers.serialize(self.G_initializer)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_data_chained(nof_total_inputs, nof_inputs_NN):
    trX = np.random.normal(10, 0.5, (nof_total_inputs, nof_inputs_NN))
    trY = np.apply_along_axis(lambda a: (a[0] + a[1]) + (a[2] + a[3]), 1, trX)
    trY = trY.reshape(nof_total_inputs,1)
    teX = np.random.normal(10, 0.5, (nof_total_inputs, nof_inputs_NN))
    teY = np.apply_along_axis(lambda a: (a[0] + a[1]) + (a[2] + a[3]), 1, teX)
    teY = teY.reshape(nof_total_inputs,1)
    return (trX, trY), (teX, teY)


def chain_3_ops():
    a_b_c_d = Input((4,))
    a_b, c_d = Lambda(lambda x: x[:, :2], output_shape=(2,))(a_b_c_d), Lambda(lambda x: x[:, 2:], output_shape=(2,))(
        a_b_c_d)

    a_b_res = NALU(1,
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(a_b)
    c_d_res = NALU(1,
                 MW_initializer=RandomNormal(stddev=1),
                 G_initializer=Constant(10))(c_d)
    res = NALU(1,
                 MW_initializer=RandomNormal(stddev=1),
                 G_initializer=Constant(10))(concatenate([a_b_res,c_d_res],axis=1))
    return Model(a_b_c_d, res)


if __name__ == "__main__":
    train = True

    if train:
        m = chain_3_ops()
        m.summary()
        m.compile("rmsprop", "mse", metrics=["mae"])
        (trX, trY), (teX, teY) = get_data_chained(2 ** 16, 4)
        K.set_value(m.optimizer.lr, 1e-2)
        m.fit(trX, trY, validation_data=(teX, teY), batch_size=1024, epochs=2000)
        K.set_value(m.optimizer.lr, 1e-3)
        m.fit(trX, trY, validation_data=(teX, teY), batch_size=1024, epochs=2000)
        K.set_value(m.optimizer.lr, 1e-4)
        m.fit(trX, trY, validation_data=(teX, teY), batch_size=1024, epochs=2000)

        #serialize weights to HDF5
        m.save_weights("weights.h5")
        print("Saved weights to disk")

    # load weights into new model
    m_trained = chain_3_ops()
    m_trained.load_weights("weights.h5")
    print("Loaded weights from disk")
    np.set_printoptions(threshold=np.inf)
    print(teX)
    print(m.predict(teX, verbose=1))