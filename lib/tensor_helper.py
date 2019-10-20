
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Lambda

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

