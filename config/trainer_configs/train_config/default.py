from yacs.config import CfgNode as CN

_C = CN()

# Define optimizer parameters
_C.optimizer = "adam"
_C.learning_rate = .001
_C.loss = "mse"

# Define training parameters
_C.batch_size = 1
_C.n_epochs = 10
_C.sample_weight_mode = "temporal"
_C.verbose = True

# Define validation parameters
_C.validation_freq = 0

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
