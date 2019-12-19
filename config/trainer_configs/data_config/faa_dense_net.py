
from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.FaaDataset import FaaDataset

_C = CN()

# Define dataset parameters
_C.dataset_name = "FaaDataset"
_C.for_training = True
_C.feature_length = 6

_C.input_shape = [_C.feature_length, 3]

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
