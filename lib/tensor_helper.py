
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Lambda
from tensorflow.keras import layers

def split(tensor, axis, keep_dims=False):

	shape = backend.int_shape(tensor)

	tensor_list = []
	for i in range(shape[axis]):

		sliced_tensor = Lambda(lambda x: x[:,i,:,:])(tensor)
		if keep_dims:
			sliced_tensor = Lambda(lambda x: backend.expand_dims(x, axis=axis))(sliced_tensor)

		tensor_list.append(sliced_tensor)

	return tensor_list

def merge(tensor_list, axis, expand_dims=False, name=""):

	if expand_dims:

		for i in range(len(tensor_list)):
			tensor_list[i] = Lambda(lambda x: backend.expand_dims(x, axis=axis))(tensor_list[i])

	merged_tensor = Lambda(lambda x: backend.concatenate(x, axis))(tensor_list)

	if name != "":
		merged_tensor = Lambda(lambda x: x, name=name)(merged_tensor)

	return merged_tensor

def map(func, tensor_list):

	func_output_list = []
	for tensor in tensor_list:
		func_output_list.append(func(tensor))

	return func_output_list

##########################################

class MyDenseStackLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, stack_config, name=""):
        if name != "":
            super(MyDenseStackLayer, self).__init__(name=name)
        else:
            super(MyDenseStackLayer, self).__init__()

        ### Collect all layers of encoder
        self.stack_layer_list = []
        for i in range(len(stack_config.n_layers_list)):
        	### Define dense layer
        	dense_layer = layers.Dense(units=stack_config.n_layers_list[i],
        		activation=stack_config.activation_type_list[i])
        	dense_layer.trainable = encoder_config.trainable_list[i]
        	self.stack_layer_list.append(dense_layer)

    def call(self, inputs):
        pipeline = inputs
        for i in range(len(self.stack_layer_list)):
            pipeline = self.stack_layer_list[i](pipeline)

        outputs = pipeline
        return outputs

class MyEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, encoder_config, name=""):
        if name != "":
            super(MyEncoderLayer, self).__init__(name=name)
        else:
            super(MyEncoderLayer, self).__init__()

        ### Collect all layers of encoder
        self.encoder_layer_list = []
        for i in range(len(encoder_config.n_filters_list)):
            ### Define convolution layer
            conv_layer = layers.Conv2D(filters=encoder_config.n_filters_list[i],
                kernel_size=encoder_config.kernel_size_list[i],
                strides=encoder_config.n_strides_list[i],
                padding=encoder_config.padding_list[i],
                activation=encoder_config.activation_type_list[i])
            conv_layer.trainable = encoder_config.trainable_list[i]
            self.encoder_layer_list.append(conv_layer)

    def call(self, inputs):
        pipeline = inputs
        for i in range(len(self.encoder_layer_list)):
            pipeline = self.encoder_layer_list[i](pipeline)

        outputs = pipeline
        return outputs

class MyDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, decoder_config, name=""):
        if name != "":
            super(MyDecoderLayer, self).__init__(name=name)
        else:
            super(MyDecoderLayer, self).__init__()

    
        self.decoder_layer_list = []
        for i in range(len(decoder_config.n_filters_list)):
            ### Define convolution layer
            transconv_layer = layers.Conv2DTranspose(filters=decoder_config.n_filters_list[i],
                kernel_size=decoder_config.kernel_size_list[i],
                strides=decoder_config.n_strides_list[i],
                padding=decoder_config.padding_list[i],
                output_padding=decoder_config.output_padding[i],
                activation=decoder_config.activation_type_list[i])
            transconv_layer.trainable = decoder_config.trainable_list[i]
            self.decoder_layer_list.append(transconv_layer)

    def call(self, inputs):
        pipeline = inputs
        for i in range(len(self.decoder_layer_list)):
            pipeline = self.decoder_layer_list[i](pipeline)

        outputs = pipeline
        return outputs

class MyNoiserLayer(tf.keras.layers.Layer):
    def __init__(self, noiser_config, name=""):
        if name != "":
            super(MyNoiserLayer, self).__init__(name=name)
        else:
            super(MyNoiserLayer, self).__init__()

        self.gaussian_noise_variable = 0
        self.binarizer_layer = 0
        self.multiplication_layer = Lambda(lambda x: x[0]+x[1])

    def call(self, inputs):
        return outputs

class MyInputLayer(tf.keras.layers.Layer):
	def __init__(self, name=""):
		if name != "":
			super(MyInputLayer, self).__init__(name=name)
		else:
			super(MyInputLayer, self).__init__()

		self.layer = Lambda(lambda x: x)

	def build(self, input_shape):
		super(MyInputLayer, self).build(input_shape)

	def call(self, inputs):
		return self.layer(inputs)

class MySplitLayer(tf.keras.layers.Layer):
	def __init__(self, axis_size):
		super(MySplitLayer, self).__init__()

		self.layer_list = []
		for i in range(axis_size):
			self.layer_list.append(Lambda(lambda x: x[:,i,:,:]))

	def build(self, input_shape):
		super(MySplitLayer, self).build(input_shape)

	def call(self, inputs):
		tensor_list = []
		for layer in self.layer_list:
			tensor_list.append(layer(inputs))
		return tensor_list

class MyMergeLayer(tf.keras.layers.Layer):

	def __init__(self, merge_axis, axis_size, expand_dims=False, name=""):
		if name != "":
			super(MyMergeLayer, self).__init__(name=name)
		else:
			super(MyMergeLayer, self).__init__()

		self.expand_dims = expand_dims
		self.axis_size = axis_size

		if expand_dims:
			self.expand_layer = Lambda(lambda x: backend.expand_dims(x, axis=merge_axis))

		self.merge_layer = Lambda(lambda x: backend.concatenate(x, merge_axis))

	def build(self, input_shape):
		super(MyMergeLayer, self).build(input_shape)

	def call(self, inputs):
		if self.expand_dims:
			for i in range(self.axis_size):
				inputs[i] = self.expand_layer(inputs[i])

		merged_inputs = self.merge_layer(inputs)

		return merged_inputs

###############################

