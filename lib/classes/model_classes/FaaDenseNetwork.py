
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers

import lib.tensor_helper as tensor_helper

class FaaDenseNetwork(Model):
    '''
    Architecture setup:
        -Dense funnel: 12(elu), 6(elu)
        -Dense past: 3(linear)
        -Dense future: 3(linear)
    Appropriate hyperparameter settings:
        -Learning rate: .001
    '''

    def __init__(self, master_config):

        data_config = master_config.Core_Config.Data_Config
        model_config = master_config.Core_Config.Model_Config

        self.Data = data_config
        self.Dense_Funnel = model_config.dense_funnel
        self.Dense_Past = model_config.dense_past
        self.Dense_Future = model_config.dense_future

        self._create_model_hooks()

    def _create_model_hooks(self):
        
        #####################
        # Layer definitions #
        #####################

        ### Input processing layers
        flatten_input = tf.layers.Flatten(name="flatten_input")

        ### Body layers
        dense_funnel = tensor_helper.MyDenseStackLayer(input_size=self.Data.input_shape[0]*self.Data.input_shape[1],
            stack_config=self.Dense_Funnel,
            name="dense_funnel")
        dense_past = layers.Dense(units=3, activation=self.Dense_Past.activation_type, name="dense_past")
        dense_future = layers.Dense(units=3, activation=self.Dense_Future.activation_type, name="dense_future")

        ### Define output processing layers
        output_past = layers.Lambda(lambda x: x, name="output_past")
        output_future = layers.Lamba(lambda x: x, name="output_future")

        #################
        # Data pipeline #
        #################
        ### Define input tensor
        input_trajectory = layers.Input(shape=(self.Data.input_shape[0], self.Data.input_shape[1]), name="input-trajectory")

        ### Flatten input
        flatten_input_out = flatten_input(input_trajectory)

        ### Pipeline body
        dense_funnel_out = dense_funnel(flatten_input_out)
        dense_past_out = dense_past(dense_funnel_out)
        dense_future_out = dense_future(dense_funnel_out)

        ### Pipeline outputs
        output_past_out = output_past(dense_past_out)
        output_future_out = output_future(dense_future_out)

        #####################
        # Initialize parent #
        #####################
        super(FaaDenseNetwork, self).__init__(inputs=[input_trajectory], outputs=[output_past_out, output_future_out])

        ###########################
        # Record callback tensors #
        ###########################
        callback_tensor_dict = {
                    "predicted_past_labels": output_past_out,
                    "predicted_future_labels": output_future_out
                }

        self.callback_tensor_dict = callback_tensor_dict





