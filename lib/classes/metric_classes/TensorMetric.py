
import tensorflow as tf
from tensorflow import keras

class TensorMetric(keras.metrics.Metric):

    def __init__(self, tensor, name, **kwargs):
      super(TensorMetric, self).__init__(name=name, **kwargs)
      self.tensor = tensor

    def update_state(self, y_true, y_pred, sample_weight=None):
      pass

    def result(self):
      return self.tensor

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      pass