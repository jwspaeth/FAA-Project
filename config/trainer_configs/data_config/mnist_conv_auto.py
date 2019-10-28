
from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset

_C = CN()

# Define dataset parameters
_C.dataset_name = "MnistAutoencoderDataset"
_C.input_shape = [28, 28, 1]

### Define label parameters
_C.Label = CN()

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
