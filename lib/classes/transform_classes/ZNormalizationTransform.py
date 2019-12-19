
import math

import numpy as np
from yacs.config import CfgNode as CN

class ZNormalizationTransform():

	def __init__(self, master_config, calibration_data):
		self.input_shape = master_config.Core_Config.Data_Config.input_shape
		
		calibration_data = np.reshape(calibration_data, (-1, self.input_shape[1]))

		if not master_config.Train_Config.transform_vars_bool:
			self.mean = np.mean(calibration_data, axis=0)
			self.stddev = np.std(calibration_data, axis=0)
		else:
			self.mean = master_config.Train_Config.transform_vars.mean
			self.stddev = master_config.Train_Config.transform_vars.stddev

	def normalize(self, data_set, label=False):
		### Return list
		normed_data_set = []

		### Loop through inputs/outputs to model
		for i, data in enumerate(data_set):
			### If label, add extra middle dimension. Only for faa data
			if label:
				data = np.expand_dims(data, axis=1)

			### Apply z score normalization
			norm_data = []
			for i in range(self.input_shape[1]):
				norm_data.append( (data[:,:,i] - self.mean[i]) / self.stddev[i] )
			norm_data = np.stack(norm_data, axis=2)
			norm_data = np.squeeze(norm_data)
			normed_data_set.append(norm_data)

		return normed_data_set

	def denormalize(self, data_set, label=False):
		### Return list
		denormed_data_set = []

		### Loop through inputs/outputs to model
		for i, data in enumerate(data_set):
			### If label, add extra middle dimension. Only for faa data
			if label:
				data = np.expand_dims(data, axis=1)

			### Invert z score normalization
			denorm_data = []
			for i in range(self.input_shape[1]):
				denorm_data.append( (data[:,:,i] * self.stddev[i]) + self.mean[i] )
			denorm_data = np.stack(denorm_data, axis=2)
			denorm_data = np.squeeze(denorm_data)
			denormed_data_set.append(denorm_data)

		return denormed_data_set

	def set_transform_variables(self, master_config):
		'''
		Set the transform variables in the master configuration object so that they save to disk after training.
		If they don't get saved to disk, future uses of the model will be incorrect.
		'''
		master_config.Train_Config.transform_vars = CN()
		master_config.Train_Config.transform_vars.mean = self.mean.tolist()
		master_config.Train_Config.transform_vars.stddev = self.stddev.tolist()

		master_config.Train_Config.transform_vars_bool = True

		return master_config




