
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers

import lib.tensor_helper as tensor_helper

class AnalysisNetwork(Model):
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
        self.Convolution = model_config.Convolution
        self.Max_Pool = model_config.Max_Pool
        self.Mean_Pool = model_config.Mean_Pool
        self.Rate = model_config.Rate
        self.Regularization = model_config.Regularization

        self._create_model_hooks()

    def _create_model_hooks(self):
        
        #####################
        # Layer definitions #
        #####################
        ### Define convolution layer
        n_strides = None
        if self.Convolution.n_strides > 0:
            n_strides = self.Convolution.n_strides
        conv_layer = layers.Conv1D(filters=self.Convolution.n_filters, kernel_size=self.Convolution.kernel_size,
                    strides=n_strides, padding=self.Convolution.padding,
                    activation=self.Convolution.activation_type,
                    kernel_regularizer=regularizers.l2(self.Regularization.l2_lambda/self.Convolution.n_filters),
                    activity_regularizer=regularizers.l1(self.Regularization.activation_lambda/self.Convolution.n_filters)
                    )

        ### Define max pooling layer
        n_strides = None
        if self.Max_Pool.n_strides > 0:
            n_strides = self.Convolution.n_strides
        max_pool_layer = layers.MaxPooling2D(pool_size=self.Max_Pool.size_input,
            strides=n_strides,padding=self.Max_Pool.padding)

        ### Define mean pool layer
        n_strides = None
        if self.Mean_Pool.n_strides > 0:
            n_strides = self.Mean_Pool.n_strides
        mean_pool_layer = layers.AveragePooling2D(pool_size=self.Mean_Pool.size_input,
                                                  strides=n_strides,
                                                  padding=self.Mean_Pool.padding)

        ### Define rate modifier layer
        rate_modifier_layer = layers.Lambda(lambda y: y * self.Rate.rate_modifier)

        ### Define output processing layer
        output_processing_layer = layers.Lambda(lambda y: backend.squeeze(y,2), name="output")

        #################
        # Data pipeline #
        #################
        ### Define input tensor
        inputs = layers.Input(shape=(self.Data.input_shape[0], self.Data.input_shape[1], self.Data.input_shape[2]), name="input-1")

        ### Split, pipe separate convolutions, and merge
        split_input = tensor_helper.split(tensor=inputs, axis=1, keep_dims=False)
        split_conv_output = tensor_helper.map(conv_layer, split_input)
        conv_output = tensor_helper.merge(tensor_list=split_conv_output, axis=1, expand_dims=True, name="conv-output") 

        ### Pipe max pool
        max_pool_out = max_pool_layer(conv_output)
        
        ### Pipe mean pool
        mean_pool_out = mean_pool_layer(max_pool_out)
        
        ### Pipe rate modifier
        mean_pool_out_mod = rate_modifier_layer(mean_pool_out)

        ### Pipe final processing
        outputs = output_processing_layer(mean_pool_out)

        super(AnalysisNetwork, self).__init__(inputs=[inputs], outputs=[outputs])

        ### Manually add dot loss to conv layer, bc keras is dumb and inflexible
        '''dot_reg = self._dot_reg_wrapper(self.Regularization.dot_lambda, self.Convolution.n_filters)
        dot_loss = dot_reg(conv_layer._trainable_weights[0])
        conv_layer.add_loss(dot_loss)

        callback_tensor_dict = {
                            "convolutional_output": conv_output,
                            "predicted_labels": outputs,
                            "convolutional_l2_loss": self._get_l2_regularizer_loss(conv_layer.losses),
                            "convolutional_activity_loss": self._get_activity_regularizer_loss(conv_layer.losses),
                            "convolutional_dot_loss": self._get_dot_regularizer_loss(conv_layer.losses)
                        }
        '''
        dot_reg = self._act_dot_reg_wrapper(self.Regularization.dot_lambda, self.Convolution.n_filters)
        dot_loss = dot_reg(conv_output)
        conv_layer.add_loss(dot_loss)

        callback_tensor_dict = {
                    "convolutional_output": conv_output,
                    "predicted_labels": outputs,
                    "convolutional_l2_loss": self._get_l2_regularizer_loss(conv_layer.losses),
                    "convolutional_activity_loss": self._get_activity_regularizer_loss(conv_layer.losses),
                    "convolutional_dot_loss": self._get_dot_regularizer_loss(conv_layer.losses)
                }

        self.callback_tensor_dict = callback_tensor_dict

    def _get_l2_regularizer_loss(self, losses):

        for t in losses:
            print(t.name)

        l2_tensor_list = [t for t in losses if "kernel/Regularizer" in t.name]
        l2_tensor = l2_tensor_list[0]
        return l2_tensor

    def _get_activity_regularizer_loss(self, losses):
        activity_tensor_list = [t for t in losses if "ActivityRegularizer" in t.name]
        activity_tensor = sum(activity_tensor_list)
        return activity_tensor

    def _get_dot_regularizer_loss(self, losses):
        dot_tensor_list = [t for t in losses if "dot_regularizer" in t.name]
        dot_tensor = dot_tensor_list[0]
        return dot_tensor

    def _act_dot_reg_wrapper(self, dot_lambda, n_filters):

        def act_dot_reg(conv_output):
            if n_filters == 1:
                return backend.variable(0, name="dot_regularizer")

            ### Create list of every filter pair combination
            filter_pair_list = []
            for i in range(0, n_filters):
                for j in range(i, n_filters):
                    filter_pair_list.append([i, j])

            ### Loop through list and feed each pair to inner product, getting a scalar for each pair. Sum these scalars
            cos_sim_list = []
            for i, pair in enumerate(filter_pair_list):
                filter_input_1 = layers.Lambda(lambda x: x[:,:,:,pair[0]])(conv_output)
                filter_input_2 = layers.Lambda(lambda x: x[:,:,:,pair[1]])(conv_output)
                cos_sim = self._act_cosine_similarity(filter_input_1, filter_input_2)

                cos_sim_list.append(cos_sim)

            ### Add all dot products to get final value
            total_sum = layers.Add()(cos_sim_list)
            total_average_sum = layers.Lambda(lambda x: x / len(filter_pair_list))(total_sum)

            ### Multiply by lambda value and rename
            final = layers.Lambda(lambda x: x * dot_lambda)(total_average_sum)
            final_squeeze = layers.Lambda(lambda x: backend.squeeze(x, axis=0), dtype='float32', name="dot_regularizer")(final)
            return final_squeeze

        return act_dot_reg

    def _act_cosine_similarity(self, conv_output_1, conv_output_2):

        dot_product_out = self._dot_product(conv_output_1, conv_output_2)

        norm_1 = self._norm(conv_output_1)
        norm_2 = self._norm(conv_output_1)

        denominator = layers.Multiply()([norm_1, norm_1])

        cos_sim_out = layers.Lambda(lambda x: x[0]/x[1])([dot_product_out, denominator])

        cos_sim_squared = layers.Lambda(lambda x: backend.square(x))(cos_sim_out)
        cos_sim_sqrt = layers.Lambda(lambda x: backend.sqrt(x))(cos_sim_squared)

        return cos_sim_sqrt

#########################################################
    def _dot_reg_wrapper(self, dot_lambda, n_filters):

        def dot_reg(weight_matrix):

            if n_filters == 1:
                return backend.variable(0, name="dot_regularizer")

            ### Create list of every filter pair combination
            filter_pair_list = []
            for i in range(0, n_filters):
                for j in range(i, n_filters):
                    filter_pair_list.append([i, j])

            ### Loop through list and feed each pair to frobenius inner product, getting a scalar for each pair. Sum these scalars
            cos_sim_list = []
            for i, pair in enumerate(filter_pair_list):
                filter_input_1 = layers.Lambda(lambda x: x[:,:,pair[0]])(weight_matrix)
                filter_input_2 = layers.Lambda(lambda x: x[:,:,pair[1]])(weight_matrix)
                cos_sim = self._cosine_similarity(filter_input_1, filter_input_2)

                cos_sim_list.append(cos_sim)

            ### Add all dot products to get final value
            total_sum = layers.Add()(cos_sim_list)
            total_average_sum = layers.Lambda(lambda x: x / len(filter_pair_list))(total_sum)

            ### Multiply by lambda value and rename
            final = layers.Lambda(lambda x: x * dot_lambda)(total_average_sum)
            final_squeeze = layers.Lambda(lambda x: backend.squeeze(x, axis=0), dtype='float32', name="dot_regularizer")(final)
            return final_squeeze

        return dot_reg

    def _cosine_similarity(self, weight_matrix_1, weight_matrix_2):

        dot_product_out = self._dot_product(weight_matrix_1, weight_matrix_2)

        norm_1 = self._norm(weight_matrix_1)
        norm_2 = self._norm(weight_matrix_2)

        denominator = layers.Multiply()([norm_1, norm_1])

        cos_sim_out = layers.Lambda(lambda x: x[0]/x[1])([dot_product_out, denominator])

        cos_sim_squared = layers.Lambda(lambda x: backend.square(x))(cos_sim_out)
        cos_sim_sqrt = layers.Lambda(lambda x: backend.sqrt(x))(cos_sim_squared)

        return cos_sim_sqrt

    def _dot_product(self, weight_matrix_1, weight_matrix_2):

        mult_out = layers.Multiply()([weight_matrix_1, weight_matrix_2])
        sum_out = layers.Lambda(lambda x: backend.sum(x))(mult_out)

        return sum_out

    def _norm(self, weight_matrix):

        square_out = layers.Lambda(lambda x: backend.square(x))(weight_matrix)
        sum_out = layers.Lambda(lambda x: backend.sum(x))(square_out)
        sqrt_out = layers.Lambda(lambda x: backend.sqrt(x))(sum_out)
        sqrt_expanded = layers.Lambda(lambda x: backend.expand_dims(x))(sqrt_out)

        return sqrt_expanded







