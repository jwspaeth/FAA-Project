
import os
import copy

import numpy as np
import pandas as pd


class FaaDataset():
	"""
	Represents the faa data.
	"""

	dataset_path = "data/faa-data/"

	def __init__(self, master_config):

		self.master_config = master_config

	def get_full_generator(self, batch_size=1):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		yield None

	def get_training_generator(self, batch_size=1, sample_weight_mode=None):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		#x_train = np.where(x_train > self.threshold_value, 1, 0)
		yield np.expand_dims(x_train, axis=3), np.expand_dims(x_train, axis=3), None

	def get_validation_generator(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		#x_test = np.where(x_test > self.threshold_value, 1, 0)
		yield np.expand_dims(x_test, axis=3), np.expand_dims(x_test, axis=3), None

	#Last three functions returns full dataset
	def get_training_features_and_classes(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		#x_test = np.where(x_test > self.threshold_value, 1, 0)
		return np.expand_dims(x_train, axis=3), y_train

	def get_validation_features_and_classes(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		#x_train = np.where(x_train > self.threshold_value, 1, 0)
		return np.expand_dims(x_test, axis=3), y_test

	def get_full_features_and_classes(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = np.expand_dims(x_train, axis=3)
		x_test = np.expand_dims(x_test, axis=3)
		x_full = np.concatenate((x_train, x_test), axis=0)
		y_full = np.concatenate((y_train, y_test), axis=0)
		#x_full = np.where(x_full > self.threshold_value, 1, 0)
		return x_full, y_full

	def _get_sample_weights(self, model_config):
		return None






		