
#####
# Fix data length issue.
####
# • Should serve week data with 0s during weeks where data is missing



import os
import copy

import numpy as np
import pandas as pd

from lib.classes.dataset_classes.BabyDataset import BabyDataset
from lib.classes.TargetRate import TargetRate

class SubjectDataset(BabyDataset):
	"""
	Represents a single subject's data. This is currently all we need in terms of functionality. Will probably
		be extended in the future
	"""

	def __init__(self, master_config=None, subject_index=""):

		if master_config is not None:
			self.master_config = master_config

		if subject_index is not "":
			self.subject_index = subject_index
		elif master_config is not None:
			self.subject_index = master_config.Core_Config.Data_Config.subject_index

		self.valid_weeks = self._create_valid_week_array()
		self.valid_mask = self._create_valid_mask_array()

		super().__init__()

	def get_full_generator(self, batch_size=1):
		data_config = self.master_config.Core_Config.Data_Config
		model_config = self.master_config.Core_Config.Model_Config

		feature_names = data_config.feature_names
		for_training = data_config.for_training

		while True:

			feature_batch = self._get_feature_batch(feature_names, for_training)
			label_batch = self._get_label_batch(data_config, model_config)

			yield (feature_batch, label_batch)

	def get_training_generator(self, batch_size=1, sample_weight_mode=None):

		data_config = self.master_config.Core_Config.Data_Config
		model_config = self.master_config.Core_Config.Model_Config

		feature_names = data_config.feature_names
		for_training = data_config.for_training

		while True:

			feature_batch = self._get_feature_batch(feature_names, for_training)
			label_batch = self._get_label_batch(data_config, model_config)
			sample_weights = self._get_sample_weights(model_config)

			if sample_weight_mode is not None:
				yield (feature_batch, label_batch, sample_weights)
			else:
				yield (feature_batch, label_batch)

	def get_validation_generator(self):
		return None

	def _get_feature_batch(self, feature_names, for_training):
		### Get feature batch
		valid_weeks_pop_list = copy.deepcopy(self.valid_weeks)

		week_list = []
		for mask_value in self.valid_mask:

			if mask_value:
				subject_location = self._create_week_filename(valid_weeks_pop_list.pop(0))
				subject_dataframe = pd.read_csv(subject_location)

				if feature_names == "all":
					subject_np = subject_dataframe.to_numpy()
					week_list.append(subject_np)
				else:
					subset_dataframe = subject_dataframe[feature_names]
					subset_np = subset_dataframe.to_numpy()
					week_list.append(subset_np)
			else:
				if feature_names == "all":
					nan_week = np.zeros(shape=(self.get_data_length(), 43))
					week_list.append(nan_week)
				else:
					nan_week = np.zeros(shape=(self.get_data_length(), len(feature_names)))
					week_list.append(nan_week)

		week_stack = np.stack(week_list)

		if for_training:
			week_stack = np.expand_dims(week_stack, axis=0)

		feature_batch = week_stack

		return feature_batch


	def _get_label_batch(self, data_config, model_config):
		### Get label batch
		### Create target rate generator and generate labels
		target_rate = TargetRate(length=self.get_n_weeks(), offset=data_config.Label.offset,
			rate_type=data_config.Label.rate_type)
		label_batch = target_rate.generate_series(model_config.Convolution.n_filters)

		return label_batch

	def get_all_feature_names(self):
		"""
		Return a list of all feature names
		"""

		subject_location = self._create_week_filename(self.valid_weeks[0])
		subject_dataframe = pd.read_csv(subject_location)

		return list(subject_dataframe.columns)

	def get_n_weeks(self):
		"""
		Return number of weeks, valid and invalid
		"""
		final_week_index = self.valid_weeks[len(self.valid_weeks)-1]
		return final_week_index

	def get_total_seconds(self):
		"""
		Return total seconds measured in data
		"""
		return 300

	def get_data_length(self):
		"""
		Return length of data
		"""
		subject_location = self._create_week_filename(self.valid_weeks[0])
		subject_dataframe = pd.read_csv(subject_location)
		return subject_dataframe.shape[0]

	def _get_sample_weights(self, model_config):
		"""
		Returns a numpy version of valid mask with an extra dimension to represent the sample index within a batch
		"""

		sample_weights = np.asarray(self.valid_mask)
		sample_weights = np.expand_dims(sample_weights, axis=0)

		return sample_weights

	def _create_valid_mask_array(self):
		"""
		Create a mask array where the valid weeks are 1 and the invalid weeks are 0
		"""

		valid_mask = []

		valid_weeks = self._create_valid_week_array()

		valid_mask = [0]*valid_weeks[len(valid_weeks)-1]
		for week_index in valid_weeks:
			week_index -= 1
			valid_mask[week_index] = 1

		return valid_mask

	def _create_valid_week_array(self):
		"""
		Create array of valid weeks by their number listed in the filesystem
		"""
		all_filenames = os.listdir(BabyDataset.data_location)

		subject_filenames = [filename for filename in all_filenames if self.subject_index in filename]

		valid_week_array = []
		while len(subject_filenames) > 0:
			week_index = self._filter_week_index(subject_filenames.pop(0))
			valid_week_array.append(week_index)

		valid_week_array.sort()
		return valid_week_array

	def _filter_week_index(self, filename):
		"""
		Filter filename and get week index
		"""
		filename = filename.replace("subject_{}_w".format(self.subject_index), "")
		filename = filename.replace(".csv", "")

		while filename[0] == "0":
			filename = self._remove_string_index(0, filename)

		return int(filename)

	def _unfilter_week_index(self, week_index):
		"""
		Unfilter week index
		"""

		week_index_str = ""
		if week_index < 10:
			week_index_str = str(week_index)
			week_index_str = self._insert_string_index(0, week_index_str, "0")
		else:
			week_index_str = str(week_index)

		return week_index_str

	def _remove_string_index(self, i, string):
		"""
		Remove string index
		"""
		return string[:i] + string[i+1:]

	def _insert_string_index(self, i, string, insert_string):
		"""
		Insert at string index
		"""

		return string[:i] + insert_string + string[i:]


	def _create_week_filename(self, week_index):
		"""
		Create subject filename for specific week
		"""
		week_index_str = self._unfilter_week_index(week_index)
		filename = BabyDataset.data_location + "subject_{}_w{}.csv".format(self.subject_index, week_index_str)

		return filename




