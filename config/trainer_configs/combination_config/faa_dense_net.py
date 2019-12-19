import importlib
from itertools import product

from yacs.config import CfgNode as CN

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset

def create_cfg_set(flag_dict, default_dict):

    ### Create master cfg
    master_cfg = CN()

    ### Get core cfg and merge with master cfg
    core_config_module = importlib.import_module("config.trainer_configs.core_config." + default_dict["cfg"])
    get_core_cfg = getattr(core_config_module, "get_cfg_defaults")
    master_cfg.Core_Config = get_core_cfg(default_dict)

    master_cfg.type = "experiment"

    train_config_module = importlib.import_module("config.trainer_configs.train_config." + default_dict["cfg"])
    get_train_cfg = getattr(train_config_module, "get_cfg_defaults")
    master_cfg.Train_Config = get_train_cfg()

    ### Options
    option_dictionary = {}

    ### Create list of combos from the option dictionary
    combo_list = None
    if option_dictionary:
        combo_list = my_product(option_dictionary)

    ### If parent process, print the option dictionary to be used and each combo list
    if not flag_dict["Is_Subprocess"]:
        print("\nOption dictionary: {}".format(option_dictionary))

        if combo_list is None:
            print("Experiment is with default parameters")
        else:
            for i, comb in enumerate(combo_list):
                print("Experiment {} option selection: {}".format(i, comb))
            print()
    ### If subprocess, print the associated combo list
    else:
        subprocess_num = flag_dict["Subprocess_Number"]
        if combo_list is None:
            print("Experiment is with default parameters")
        else:
            print("\nExperiment {} option selection: {}\n".format(subprocess_num, combo_list[subprocess_num]))

    ### Iteration over combo list and return the master config set
    cfg_set = []
    if combo_list is None:
        subprocess_num = flag_dict["Subprocess_Number"]
        master_cfg.iteration_parameters = "Default; Check configuration files for relevant model."
        master_cfg.child_name = "experiment_{}".format(subprocess_num)
        cfg_set.append(master_cfg)
    else:
        for i in range(len(combo_list)):

            current_combo = combo_list[i]
            current_mcfg = master_cfg.clone()

            ### Create config parameter to hold experiment parameters, accessible for recording purposes only
            current_mcfg.iteration_parameters = [current_combo]

            # Must be modified to when option dictionary is modified
            #current_mcfg.Core_Config.Model_Config.Regularization.dot_lambda = current_combo["dot_lambda"]

            subprocess_num = flag_dict["Subprocess_Number"]
            if flag_dict["Experiment"]:
                current_mcfg.child_name = "experiment_{}".format(subprocess_num)
            else:
                current_mcfg.child_name = "evaluation_{}".format(subprocess_num)

            cfg_set.append(current_mcfg)

    return cfg_set

### From stack overflow
def my_product(inp):
    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]

def get_cfg_set(flag_dict, default_dict):
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern

  cfg_set = create_cfg_set(flag_dict, default_dict)

  return cfg_set