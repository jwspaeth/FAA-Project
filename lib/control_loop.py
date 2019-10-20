
import sys
import importlib
import os
import subprocess
import time
import copy

from lib.trainer import train

def control_loop():
    """
    Method
        Creates and runs the experiments or evaluations defined in the configuration folders

    Command Line Args (Exclude quotation marks when using)
        "-p": Run in parallel. This is only recommended on the supercomputer, as it will be able to handle the parallel jobs. Standard computers will likely struggle
        "-super": Running on supercomputer. Uses supercomputer batch job script to spawn subprocesses rather than standard bash scripts. Default is standard bash script
        "-data_default=[input]": Define an alternate default configuration file for data.
        "-save_default=[input]": Define an alternate default configuration file for saving.
        "-model_default=[input]": Define an alternate default configuration file for model.
        "-master_default=[input]": Define an alternate default configuration file for the master configuration file.
        "-experiment_default=[input]": Define an alternate default configuration file for the experiment set constructor.
        "-evaluation_default=[input]": Define an alternate default configuration file for the evaluation set constructor
    """

    #####################################################
    ### Create order of operations for entire program ###
    #####################################################

    ### Parse arguments for experiment or evaluation
    flag_dict, default_dict = parse_args()

    ### Create test batch path
    test_batch_path = get_test_batch_path(flag_dict, default_dict)

    ### Is this the parent process or a subprocess?
    if flag_dict["Is_Subprocess"]:

        ### Grab subprocess number
        subprocess_num = flag_dict["Subprocess_Number"]

        ### Redirect stdout and stderr based on subprocess number
        if not os.path.exists(test_batch_path + "experiment_{}/".format(subprocess_num)):
            os.mkdir(test_batch_path + "experiment_{}/".format(subprocess_num))
        log_file = open(test_batch_path + "experiment_{}/experiment_log.txt".format(subprocess_num), "w")
        sys.stdout = log_file
        sys.stderr = log_file

        ### Get config batch for experiment or evaluation
        cfgs = get_configs(flag_dict, default_dict)

        ### Grab current master config based on subprocess number
        current_cfg = cfgs[subprocess_num]

        ### Is this an experiment or an evaluation?
        if flag_dict["Experiment"]:
            train(master_config=current_cfg)

    else:
        
        ### Redirect parent stdout and stderr to file
        if not os.path.exists(test_batch_path):
            os.mkdir(test_batch_path)
        log_file = open(test_batch_path + "parent_log.txt", "w")
        sys.stdout = log_file
        sys.stderr = log_file

        ### Get config batch for experiment or evaluation
        cfgs = get_configs(flag_dict, default_dict)

        ### Spawn subprocess
        spawn_and_manage_subprocesses(n_subprocesses=len(cfgs), flag_dict=flag_dict, test_batch_path=test_batch_path)




##################
# Helper Methods #
##################

def spawn_and_manage_subprocesses(n_subprocesses, flag_dict, test_batch_path):

    ### Create subprocesses
    running_processes = []
    
    ### Feed experiment or evaluation flag to the subprocess
    arg_type = ""
    if flag_dict["Experiment"]:
        arg_type = "-x"

    ### Script to run depends on whether or not this is a supercomputer or standard computer
    script_to_run = ""
    if flag_dict["Is_Supercomputer"]:
        script_to_run = ["sbatch", "supercomputer-batch-job.sh"]
    else:
        script_to_run = ["./standard-job.sh"]

    ### Loop to spawn all the subprocesses
    print("Number of subprocesses: {}".format(n_subprocesses))
    print("Parallel flag: {}".format(flag_dict["Is_Parallel"]))
    iteration_start_time = time.time()
    for i in range(n_subprocesses):

        ### Run the subprocess and redirect stdout and stderr to log file
        print("Starting subprocess {}...\n".format(i), flush=True)
        process = subprocess.Popen([*script_to_run, arg_type, "-subprocess_number={}".format(i)])
        subprocess_start_time = time.time()
        
        ### If this isn't parallel, wait for the process to finish before continuing the loop
        if not flag_dict["Is_Parallel"]:
            process.wait()
            formatted_runtime = get_formatted_runtime(subprocess_start_time, time.time())
            print("\tSubprocess {} is finished! Runtime: {} secs\n".format(i, formatted_runtime), flush=True)
        else:
            running_processes.append((i, process, subprocess_start_time))

    ### Watch running processes and print when they're finished
    check_running_processes(running_processes=running_processes)

    formatted_runtime = get_formatted_runtime(iteration_start_time, time.time())
    print("Subprocesses finished! Runtime: {} secs".format(formatted_runtime), flush=True)

def check_running_processes(running_processes):

    ### While there are processes still running
    while len(running_processes) > 0:

        ### Loop through each remaining process
        for process_tuple in running_processes:

            ### If the process is finished
            if process_tuple[1].poll() == 0:
                formatted_runtime = get_formatted_runtime(process_tuple[2], time.time())
                print("\tSubprocess {} is finished! Runtime: {} secs\n".format(process_tuple[0], formatted_runtime), flush=True)
                running_processes.remove(process_tuple)

        ### Only check every 5 seconds
        time.sleep(5)

def get_formatted_runtime(start_time, end_time):
    runtime = end_time - start_time
    runtime = round(runtime, 2)
    return runtime

def parse_args():

    ### Flag dictionary with various flags relevant to the program
    flag_dict = {
        "Experiment": True,
        "Is_Parallel": False,
        "Is_Subprocess": False,
        "Subprocess_Number": None,
        "Is_Supercomputer": False
    }

    ### Declare default dictionary of default configuration files for every configuration type
    default_dict = {
        "data_default": "default",
        "save_default": "default",
        "model_default": "default",
        "core_default": "default",
        "train_default": "default",
        "evaluation_default": "default",
        "combination_default": "default",
        "reload_default": "default"
    }

    if len(sys.argv) > 1:

        #### Determine if this is going to be a parallel operation
        if "-p" in sys.argv:
            flag_dict["Is_Parallel"] = True

        ### Determine if this is currently a subprocess and if so, what the subprocess number is
        for arg in sys.argv:
            if "-subprocess_number=" in arg:
                flag_dict["Is_Subprocess"] = True
                flag_dict["Subprocess_Number"] = int(arg.replace("-subprocess_number=", ""))

        ### Determine if this is running on a supercomputer, which implies a batch job system
        if "-super" in sys.argv:
            flag_dict["Is_Supercomputer"] = True

        ### Determine if alternative default configuration files are indicated
        for arg in sys.argv:
            default_keys = list(default_dict.keys())
            for key in default_keys:
                if key in arg:
                    default_dict[key] = arg.replace("-"+key+"=", "")

    return flag_dict, default_dict

def get_configs(flag_dict, default_dict):

    ### Get config set
    import_string = ""
    if flag_dict["Experiment"]:
        import_string = "config.trainer_configs.combination_config." + default_dict["combination_default"]

    config_module = importlib.import_module(import_string)
    config_method = getattr(config_module, "get_cfg_set")

    return config_method(flag_dict, default_dict)

def get_test_batch_path(flag_dict, default_dict):

    ### Get save config
    import_string = ""
    if flag_dict["Experiment"]:
        import_string = "config.trainer_configs.save_config." + default_dict["save_default"]

    config_module = importlib.import_module(import_string)
    config_method= getattr(config_module, "get_cfg_defaults")

    save_config = config_method()
    test_batch_path = "results/" + save_config.Output.batch_name + "/"
    return test_batch_path


def set_experiment_flag(flag_dict):
    flag_dict["Experiment"] = True

