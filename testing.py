#!/usr/bin/env python3

from yacs.config import CfgNode as CN
from yacs.config import load_cfg

with open("results/test_batch/experiment_0/model_and_config_folder/master_config", "rt") as cfg_file:
	cfg = load_cfg(cfg_file)
	print(cfg)