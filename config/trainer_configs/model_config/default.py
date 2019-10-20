
from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset
from config.model_helper import get_size_input

_C = CN()

_C.framework = "Keras"
_C.model_type = "AnalysisNetwork"
_C.build_type = "subclass"

# Define convolutional parameters
_C.Convolution = CN()
_C.Convolution.n_filters = 1
_C.Convolution.kernel_size = 25
_C.Convolution.activation_type = "sigmoid"
_C.Convolution.n_strides = 1
_C.Convolution.padding = "same"


# Define max pool parameters
_C.Max_Pool = CN()
_C.Max_Pool.size_in_seconds = 10
subject = SubjectDataset(subject_index="c2")
_C.Max_Pool.size_input = (1, get_size_input(conv_config=_C.Convolution,
    max_pool_config=_C.Max_Pool, total_steps=subject.get_data_length(), total_seconds=subject.get_total_seconds()))
_C.Max_Pool.n_strides = 0
_C.Max_Pool.padding = "same"

# Define mean pool parameters
_C.Mean_Pool = CN()
_C.Mean_Pool.size_input = _C.Max_Pool.size_input
_C.Mean_Pool.n_strides = 0
_C.Mean_Pool.padding = "same"

# Define rate parameters
_C.Rate = CN()
_C.Rate.rate_modifier = 2

# Define regularization parameters
_C.Regularization = CN()
_C.Regularization.l2_lambda = .000001
_C.Regularization.activation_lambda = .00001
_C.Regularization.dot_lambda = .1

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
