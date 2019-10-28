
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend
from tensorflow.keras import regularizers
import tensorflow as tf

import lib.tensor_helper as tensor_helper

class GeometricAutoencoder(Model):
    '''
    Appropriate hyperparameter combos:
        learning rate: .001
        n_filters: 1
            kernel l2 lambda: 1e-6
            activation lambda: 1e-5
        n_filters: 2
            kernel l2 lambda: ?
            activation lambda: ?

    '''

    def __init__(self, master_config):

        data_config = master_config.Core_Config.Data_Config
        model_config = master_config.Core_Config.Model_Config

        self.Data = data_config
        self.Encoder = model_config.Encoder
        self.GeoDecoder = model_config.GeoDecoder
        self.Noiser = model_config.Noiser
        self.GeoEncoder = model_config.GeoEncoder
        self.Decoder = model_config.Decoder
        self.Regularization = model_config.Regularization

        self._create_model_hooks()

    def _create_model_hooks(self):
        
        #####################
        # Layer definitions #
        #####################

        encoder = tensor_helper.MyEncoderLayer(self.Encoder, name="encoder")
        geo_decoder = tensor_helper.MyDecoderLayer(self.GeoDecoder, name="geo_decoder")
        binarizer = Lambda(lambda x: self._binary_activation(x))
        noiser = layers.GaussianNoise(stddev=1)
        geo_encoder = tensor_helper.MyEncoderLayer(self.GeoEncoder, name="geo_encoder")
        decoder = tensor_helper.MyDecoderLayer(self.Decoder, name="decoder")

        ### Define output processing layer
        output_processing_layer = layers.Lambda(lambda x: x, name="output")

        #################
        # Data pipeline #
        #################
        ### Define input tensor
        inputs = layers.Input(shape=(self.Data.input_shape[0], self.Data.input_shape[1], self.Data.input_shape[2]), name="input-0")

        encoder_out = encoder(inputs)
        geo_decoder_out = geo_decoder(encoder_out)
        #binarizer_out = binarizer(geo_decoder_out)
        noiser_out = noiser(geo_decoder_out)
        geo_encoder_out = geo_encoder(noiser_out)
        decoder_out = decoder(geo_encoder_out)
        outputs = output_processing_layer(decoder_out)

        latent_space_loss = self._create_latent_space_loss(encoder_out, geo_encoder_out)
        geo_encoding_loss = self._create_geo_encoding_loss(geo_decoder_out)
        output_processing_layer.add_loss(latent_space_loss)
        output_processing_layer.add_loss(geo_encoding_loss)

        #####################
        # Initialize parent #
        #####################
        super(GeometricAutoencoder, self).__init__(inputs=[inputs], outputs=[outputs])

        ###########################
        # Record callback tensors #
        ###########################
        callback_tensor_dict = {
                    "latent_space_loss": latent_space_loss,
                    "geo_encoding_loss": geo_encoding_loss,
                    "latent_vector_1": encoder_out,
                    "latent_vector_2": geo_encoder_out,
                    "geometric_encoding": geo_decoder_out,
                    "predicted_labels": outputs
                }

        self.callback_tensor_dict = callback_tensor_dict

    def _create_latent_space_loss(self, latent_space_1, latent_space_2):
        subtract_out = layers.Subtract()([latent_space_1, latent_space_2])
        power_out = layers.Lambda(lambda x: backend.pow(subtract_out, 2))(subtract_out)
        sum_out = layers.Lambda(lambda x: backend.sum(power_out, axis=[1,2,3]))(power_out)
        mean_out = layers.Lambda(lambda x: backend.mean(sum_out, axis=0))(sum_out)
        reg_out = layers.Lambda(lambda x: x * self.Regularization.latent_space_lambda)(mean_out)
        identity_out = layers.Lambda(lambda x: x, name="latent_space_loss")(reg_out)

        return identity_out

    def _create_geo_encoding_loss(self, geo_encoding):
        abs_out = Lambda(lambda x: backend.abs(x))(geo_encoding)
        sum_out = Lambda(lambda x: backend.sum(x))(abs_out)
        identity_out = Lambda(lambda x: x*self.Regularization.geo_encoding_lambda, name="geo_encoding_loss")(sum_out)

        return identity_out

    def _binary_activation(self, x):

        output = Lambda(lambda x: 1/(1+backend.exp(-1*self.GeoDecoder.threshold_constant*x)))(x)

        return output


