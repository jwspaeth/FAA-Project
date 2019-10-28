
from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset
from config.model_helper import get_size_input

_C = CN()

_C.framework = "Keras"
_C.model_type = "ConvAutoencoder"
_C.build_type = "subclass"

# Define encoder parameters
_C.Encoder = CN()
_C.Encoder.n_filters_list = [32, 32, 16, 16, 8, 8]
_C.Encoder.kernel_size_list = [(3, 3), (2, 2), (3, 3), (2, 2), (3, 3), (2, 2)]
_C.Encoder.activation_type_list = ["elu", "elu", "elu", "elu", "elu", "sigmoid"]
_C.Encoder.n_strides_list = [(2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)]
_C.Encoder.padding_list = ["same", "same", "same", "same", "same", "same"]

# Define encoder parameters
_C.Decoder = CN()
_C.Decoder.n_filters_list = [8, 16, 32, 1]
_C.Decoder.kernel_size_list = [(2, 2), (3, 3), (3, 3), (2, 2)]
_C.Decoder.activation_type_list = ["elu", "elu", "elu", "elu", "elu", "sigmoid"]
_C.Decoder.n_strides_list = [(2, 2), (2, 2), (2, 2), (1,1)]
_C.Decoder.padding_list = ["same", "same", "same", "same", "same", "same"]
_C.Decoder.output_padding = [(1, 1), None, None, None]

# Define noise parameters
_C.Noise = CN()
_C.Noise.dummy_val = 0

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
