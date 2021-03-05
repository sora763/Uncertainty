from __future__ import absolute_import

from tensorflow.keras.layers import Dropout
import tensorflow.keras.backend as K
import warnings
from tensorflow.python.keras.layers import Lambda


class MCDropout(Dropout):
    """ Drops elements of input variable randomly.

    See the paper by Y. Gal, and G. Zoubin: `Dropout as a bayesian approximation: \
    Representing model uncertainty in deep learning .\
    <https://arxiv.org/abs/1506.02142>`
    """

    def call(self, inputs, training=None):

        if training is not None:
            if not training:
                warnings.warn('Training option is ignored..')

        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            #def dropped_inputs():
                #return Lambda(lambda x:K.dropout(inputs, self.rate, noise_shape,
                                 #seed=self.seed))
            #return dropped_inputs

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, dropped_inputs,
                                    training=True) # NOTE: force
        return inputs
