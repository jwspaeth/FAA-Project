from yacs.config import CfgNode as CN

_C = CN()

# Define output parameters
_C.Output = CN()
_C.Output.batch_name = "FaaDense-test-2"
_C.Output.checkpoint_trigger = 1

# Define callback parameters
_C.Callback = CN()
_C.Callback.exists = True
_C.Callback.names = ["MyRecordingCallback", "MemoryCallback", "ResetHistoryCallback"]
_C.Callback.figwidth = 12
_C.Callback.figheight = 2

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
