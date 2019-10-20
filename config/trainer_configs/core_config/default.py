import importlib

from yacs.config import CfgNode as CN

def create_master_config(default_dict):

	data_config_module = importlib.import_module("config.trainer_configs.data_config." + default_dict["data_default"])
	save_config_module = importlib.import_module("config.trainer_configs.save_config." + default_dict["save_default"])
	model_config_module = importlib.import_module("config.trainer_configs.model_config." + default_dict["model_default"])
	reload_config_module = importlib.import_module("config.trainer_configs.reload_config." + default_dict["reload_default"])

	get_data_cfg = getattr(data_config_module, "get_cfg_defaults")
	get_save_cfg = getattr(save_config_module, "get_cfg_defaults")
	get_model_cfg = getattr(model_config_module, "get_cfg_defaults")
	get_reload_cfg = getattr(reload_config_module, "get_cfg_defaults")

	_C = CN()

	# Load data parameters
	_C.Data_Config = get_data_cfg()

	# Load save parameters
	_C.Save_Config = get_save_cfg()

	# Define convolutional parameters
	_C.Model_Config = get_model_cfg()

	# Define reload parameters
	_C.Reload_Config = get_reload_cfg()

	return _C.clone()


def get_cfg_defaults(default_dict):
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern

  _C = create_master_config(default_dict)

  return _C.clone()