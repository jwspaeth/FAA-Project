
from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset
from config.model_helper import get_size_input

_C = CN()

_C.framework = "Keras"
_C.model_type = "GeometricAutoencoder"
_C.build_type = "subclass"

trainable_freeze = False

# Define encoder parameters
_C.Encoder = CN()
_C.Encoder.n_filters_list = [32, 32, 16, 16, 8, 8]
_C.Encoder.kernel_size_list = [(3, 3), (2, 2), (3, 3), (2, 2), (3, 3), (2, 2)]
_C.Encoder.activation_type_list = ["elu", "elu", "elu", "elu", "elu", "sigmoid"]
_C.Encoder.n_strides_list = [(2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)]
_C.Encoder.padding_list = ["same", "same", "same", "same", "same", "same"]
_C.Encoder.trainable_list = [trainable_freeze, trainable_freeze, trainable_freeze, trainable_freeze, trainable_freeze,
					trainable_freeze
					]

# Define geo decoder parameters
_C.GeoDecoder = CN()
_C.GeoDecoder.n_filters_list = [8, 16, 32, 32, 1, 1]
_C.GeoDecoder.kernel_size_list = [(2, 2), (3, 3), (3, 3), (2, 2), (2,2), (3, 3)]
_C.GeoDecoder.activation_type_list = ["elu", "elu", "elu", "elu", "elu", "sigmoid"]
_C.GeoDecoder.n_strides_list = [(2, 2), (2, 2), (2, 2), (2,2), (2, 2), (1, 1)]
_C.GeoDecoder.padding_list = ["same", "same", "same", "same", "same", "same"]
_C.GeoDecoder.output_padding = [(1, 1), None, None, None, None, None]
_C.GeoDecoder.trainable_list = [True]*6
_C.GeoDecoder.threshold_constant = 1000

# Define geo encoder parameters
_C.GeoEncoder = CN()
_C.GeoEncoder.n_filters_list = [32, 32, 16, 8, 8]
_C.GeoEncoder.kernel_size_list = [(3, 3), (3, 3), (3, 3), (3, 3), (3,3)]
_C.GeoEncoder.activation_type_list = ["elu", "elu", "elu", "elu", "sigmoid"]
_C.GeoEncoder.n_strides_list = [(2, 2), (2, 2), (2, 2), (2,2), (2,2)]
_C.GeoEncoder.padding_list = ["same", "same", "same", "same", "same"]
_C.GeoEncoder.trainable_list = [True]*5

# Define encoder parameters
_C.Decoder = CN()
_C.Decoder.n_filters_list = [8, 16, 32, 1]
_C.Decoder.kernel_size_list = [(2, 2), (3, 3), (3, 3), (2, 2)]
_C.Decoder.activation_type_list = ["elu", "elu", "elu", "elu", "elu", "sigmoid"]
_C.Decoder.n_strides_list = [(2, 2), (2, 2), (2, 2), (1,1)]
_C.Decoder.padding_list = ["same", "same", "same", "same", "same", "same"]
_C.Decoder.output_padding = [(1, 1), None, None, None]
_C.Decoder.trainable_list = [trainable_freeze, trainable_freeze, trainable_freeze, trainable_freeze, trainable_freeze,
					trainable_freeze
					]

# Define noise parameters
_C.Noiser = CN()
_C.Noiser.dummy_val = 0

# Regularization paramters
_C.Regularization = CN()
_C.Regularization.latent_space_lambda = 1000
_C.Regularization.geo_encoding_lambda = .001

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
