
import os
import importlib
import tracemalloc
import copy

import numpy as np
from tensorflow.keras import optimizers
import yaml
from yacs.config import load_cfg
import tensorflow as tf
from tensorflow import keras

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset
from lib.classes.TargetRate import TargetRate
from lib.classes.metric_classes.TensorMetric import TensorMetric

def train(master_config):

    tf.compat.v1.disable_eager_execution()

    print("Training!", flush=True)

    ### Reload old master config if applicable
    if master_config.Core_Config.Reload_Config.reload:
        old_mcfg = reload_config(master_config.Core_Config.Reload_Config.reload_path)
        if not master_config.Core_Config.Reload_Config.by_name:
            master_config.Core_Config.Model_Config = old_mcfg.Core_Config.Model_Config.clone()

    ### Build model based on master config
    model = build_model(master_config)
    if not model:
        return False

    ### Unload configurations
    data_config = master_config.Core_Config.Data_Config
    model_config = master_config.Core_Config.Model_Config
    save_config = master_config.Core_Config.Save_Config
    train_config = master_config.Train_Config

    ### Generate sample weighting
    sample_weight_mode = None
    if train_config.sample_weight_mode is not False:
        sample_weight_mode = train_config.sample_weight_mode

    ### Create dataset
    dataset = create_dataset(master_config)

    ### Get training generator
    training_generator = dataset.get_training_generator(
        batch_size=train_config.batch_size, sample_weight_mode=sample_weight_mode)

    ### Get validation generator
    validation_generator = dataset.get_validation_generator()

    ### Generate optimizer and compile model
    optimizer = None
    if train_config.optimizer == "adam":
        optimizer = optimizers.Adam(lr=train_config.learning_rate)

    model.compile(optimizer=train_config.optimizer,
                 loss=train_config.loss, sample_weight_mode=sample_weight_mode,
                 metrics=["mse"])

    ### Print model summary
    model.summary()

    ### Create callbacks
    callbacks = None
    if save_config.Callback.exists:
        callbacks = import_callbacks(master_config)

    ### Import training data
    features_train, labels_train, __ = next(training_generator)

    ### Calibrate transforms on training data, add to model, then apply to training data
    if train_config.transform:

        ### Create transform
        model.feature_transform = import_transform(master_config, features_train)

        ### Save transform variables to disk
        master_config = model.feature_transform.set_transform_variables(master_config)

        ### Apply to features
        features_train = model.feature_transform.normalize(features_train)

        ### If regression, also apply to labels.
        if train_config.regression:
            labels_train = model.feature_transform.normalize(labels_train, label=True)

    ### Import validation data
    features_validation, labels_validation, __ = next(validation_generator)

    ### Apply calibrated transforms to validation data
    if train_config.transform:
        features_validation = model.feature_transform.normalize(features_validation)

        ### If regression, also apply to labels
        if train_config.regression:
            labels_validation = model.feature_transform.normalize(labels_validation, label=True)

    validation_tuple = None
    if train_config.validation_freq > 0:
        validation_tuple = (features_validation, labels_validation)

    print("Features dimensions: {}".format(features_train[0].shape))
    print("Sample weights: {}".format(dataset._get_sample_weights()))

    if train_config.n_epochs != 0:
        '''model.fit_generator(training_generator, steps_per_epoch=1,
            validation_data=validation_generator, validation_steps=train_config.vsteps, validation_freq=train_config.vfreq,
            epochs=train_config.n_epochs, verbose=train_config.verbose, callbacks=callbacks)'''
        model.fit(
            x=features_train,
            y=labels_train,
            epochs=train_config.n_epochs,
            verbose=train_config.verbose,
            validation_data=validation_tuple,
            validation_freq=train_config.validation_freq,
            callbacks=callbacks,
            sample_weight=dataset._get_sample_weights(model_config),
            batch_size=train_config.batch_size)
    else:
        model.evaluate_generator(dataset_generator, steps=1, callbacks=callbacks, verbose=train_config.verbose)

    ### Save model after training
    save_model_and_configuration(model, master_config)

def reload_config(reload_path):
    cfg = None
    with open(reload_path + "model_and_config/master_config", "rt") as cfg_file:
        cfg = load_cfg(cfg_file)
    return cfg

def create_dataset(master_config):
    available_datasets = os.listdir("lib/classes/dataset_classes")
    available_datasets = [i.replace(".py", "") for i in available_datasets]

    dataset_name = master_config.Core_Config.Data_Config.dataset_name

    dataset_class = None
    if dataset_name in available_datasets:
        dataset_class_module = importlib.import_module("lib.classes.dataset_classes." + dataset_name)
        dataset_class = getattr(dataset_class_module, dataset_name)
    else:
        print("Error: dataset name {} not available. Check lib/classes/dataset_classes/ for available datasets".format(
            dataset_name), flush=True)
        return False

    dataset = dataset_class(master_config=master_config)

    return dataset

def import_transform(master_config, data):
    available_transforms = os.listdir("lib/classes/transform_classes")
    available_transforms = [i.replace(".py", "") for i in available_transforms]

    imported_transform = 0
    for transform_name in master_config.Train_Config.transform:

        transform_class = None
        if transform_name in available_transforms:
            transform_class_module = importlib.import_module("lib.classes.transform_classes." + transform_name)
            transform_class = getattr(transform_class_module, transform_name)
        else:
            print("Error: transform name {} not available. Check lib/classes/transform_classes/ for available transforms".format(
                transform_name), flush=True)
            return False

        transform = transform_class(master_config, data)

    return transform

def import_callbacks(master_config):
    available_callbacks = os.listdir("lib/classes/callback_classes")
    available_callbacks = [i.replace(".py", "") for i in available_callbacks]

    imported_callbacks = []
    for callback_name in master_config.Core_Config.Save_Config.Callback.names:

        callback_class = None
        if callback_name in available_callbacks:
            callback_class_module = importlib.import_module("lib.classes.callback_classes." + callback_name)
            callback_class = getattr(callback_class_module, callback_name)
        else:
            print("Error: callback name {} not available. Check lib/classes/callback_classes/ for available callbacks".format(
                callback_name), flush=True)
            return False

        callback = callback_class(master_config)
        imported_callbacks.append(callback)

    return imported_callbacks

def get_metrics(model):

    metrics_list = []

    for key, value in model.callback_tensor_dict.items():
        print("Value: {}".format(value))
        metric = TensorMetric(tensor=value, name=key)
        metrics_list.append(metric)

    return metrics_list

def save_model_and_configuration(model, master_config):

    child_name = master_config.child_name

    save_folder_path = "results/" + master_config.Core_Config.Save_Config.Output.batch_name + "/" + child_name
    save_folder_path += "/model_and_config/"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    model.save_weights(save_folder_path + "final_model_weights.h5")

    with open(save_folder_path + 'master_config', 'wt') as config_file:
        config_file.write(master_config.dump())

def feature_check(features):
    if np.isnan(features).any():
        err_str = "Nan value exists in features: {}\n".format(np.isnan(features).any())
        err_str += "Indices where nan value exists in features:\n{}".format(np.argwhere(np.isnan(features)))
        err_str += "How many nan values are in features: {}\n".format(np.argwhere(np.isnan(features)).shape)
        err_str += "Error: nan values not allowed in feature input"
        print(err_str, flush=True)
        return 0
    else:
        return 1

def label_check(labels):
    if np.isnan(labels).any():
        err_str = "Nan value exists in labels: {}\n".format(np.isnan(labels).any())
        err_str += "Indices where nan value exists in labels:\n{}".format(np.argwhere(np.isnan(labels)))
        err_str += "How many nan values are in labels: {}\n".format(np.argwhere(np.isnan(labels)).shape)
        err_str += "Error: nan values not allowed in label input"
        print(err_str, flush=True)
        return 0
    else:
        return 1

def build_model(master_config):
    """
    Imports the proper model class and builds model
    """
    available_models = os.listdir("lib/classes/model_classes")
    available_models = [i.replace(".py", "") for i in available_models]
    model_type = master_config.Core_Config.Model_Config.model_type

    model_class = None
    if model_type in available_models:
        model_class_module = importlib.import_module("lib.classes.model_classes." + model_type)
        model_class = getattr(model_class_module, model_type)
    else:
        print("Error: model type not available. Check lib/classes/model_classes/ for available models", flush=True)
        return False

    model = model_class(master_config)

    if master_config.Core_Config.Reload_Config.reload:
        reload_path = master_config.Core_Config.Reload_Config.reload_path
        if master_config.Core_Config.Reload_Config.by_name:
            model.load_weights(reload_path + "model_and_config/final_model_weights.h5", by_name=True)
        else:
            model.load_weights(reload_path + "model_and_config/final_model_weights.h5")

    return model
