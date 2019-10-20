
from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset

_C = CN()

# Define dataset parameters
_C.dataset_name = "SubjectDataset"
_C.subject_index = "k3"
_C.feature_names = ["left_wrist_x", "left_wrist_y", "left_wrist_z"]
_C.for_training = True

subject = SubjectDataset(subject_index = _C.subject_index)
_C.input_shape = [subject.get_n_weeks(), subject.get_data_length(), len(_C.feature_names)]

### Define label parameters
_C.Label = CN()
_C.Label.rate_type = "sigmoid"
_C.Label.rate_extra = "None"
_C.Label.offset = 6

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
