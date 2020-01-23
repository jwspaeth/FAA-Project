
from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset
from config.model_helper import get_size_input

_C = CN()

_C.framework = "Keras"
_C.model_type = "FaaDenseNetwork"
_C.build_type = "subclass"

# Define dense funnel
df_trainable = True
_C.Dense_Funnel = CN()
_C.Dense_Funnel.n_layers_list = [18, 9, 5]
_C.Dense_Funnel.activation_type_list = ["elu", "elu", "elu"]
_C.Dense_Funnel.trainable_list = [df_trainable]*3

# Define dense past
_C.Dense_Past = CN()
_C.Dense_Past.activation_type = "linear"

# Define dense future
_C.Dense_Future = CN()
_C.Dense_Future.activation_type = "linear"

# Define regularization parameters
_C.Regularization = CN()
_C.Regularization.dummy = 0

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
