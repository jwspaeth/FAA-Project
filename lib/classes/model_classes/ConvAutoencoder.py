
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers

import lib.tensor_helper as tensor_helper

class ConvAutoencoder(Model):
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
        self.Decoder = model_config.Decoder

        self._create_model_hooks()

    def _create_model_hooks(self):
        
        #####################
        # Layer definitions #
        #####################

        encoder = tensor_helper.MyEncoderLayer(self.Encoder, name="encoder")
        decoder = tensor_helper.MyDecoderLayer(self.Decoder, name="decoder")

        ### Define output processing layer
        output_processing_layer = layers.Lambda(lambda x: x, name="output")

        #################
        # Data pipeline #
        #################
        ### Define input tensor
        inputs = layers.Input(shape=(self.Data.input_shape[0], self.Data.input_shape[1], self.Data.input_shape[2]), name="input-0")

        encoder_out = encoder(inputs)
        decoder_out = decoder(encoder_out)
        outputs = output_processing_layer(decoder_out)

        #####################
        # Initialize parent #
        #####################
        super(ConvAutoencoder, self).__init__(inputs=[inputs], outputs=[outputs])

        ###########################
        # Record callback tensors #
        ###########################
        callback_tensor_dict = {
                    "latent_vector": encoder_out,
                    "predicted_labels": outputs
                }

        self.callback_tensor_dict = callback_tensor_dict





