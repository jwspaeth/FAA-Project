
import os

import numpy as np
import pandas as pd

class BabyDataset:
	"""
	Acts as a parent class for subject data. Fairly abstract and empty right now, but might be
		extended in the future
	"""

	### Store location of data
	data_location = "data/baby1/"

	def __init__(self):
		pass