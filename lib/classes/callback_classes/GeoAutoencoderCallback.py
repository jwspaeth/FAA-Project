
import os
import time
import resource
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
from tensorflow.keras import Model

from lib.classes.callback_classes.MyRecordingCallback import MyRecordingCallback
from lib.classes.dataset_classes.MnistAutoencoderDataset import MnistAutoencoderDataset

class GeoAutoencoderCallback(MyRecordingCallback):

    def __init__(self, master_config):
        super(GeoAutoencoderCallback, self).__init__(master_config)

        self.iteration_parameters = master_config.iteration_parameters

        self.mnist = MnistAutoencoderDataset(master_config=master_config)

    def on_train_end(self, logs):
        super().on_train_end(logs)
        self._collect_comparisons()

    def _checkpoint(self):
        super()._checkpoint()

        self._create_statistics_file()

    def _create_statistics_file(self):
        
        ### Create statistics representative of training
        ### This includes printing total loss and all separate regularization losses
        print("Saving statistics...", flush=True)
        save_file_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name
        save_file_path += "/statistics.txt"

        with open(save_file_path, "w") as f:
            f.write("Experiment parameters: {}\n".format(self.iteration_parameters))
            f.write("Epoch count: {}\n".format(self.epoch_count))
            f.write("Total loss: {}\n".format(round(self.recent_epoch_logs["loss"], 5)))

        print("Statistics saved!", flush=True)

    def _collect_comparisons(self):
        ### Collect all features, labels, and predictions
        x_test, y_test = self.mnist.get_validation_features_and_classes()
        predictions = self.model.predict(x_test)

        ### Test evaluation of tensor
        tensor_model = Model(inputs=self.model.inputs, outputs=self.model.callback_tensor_dict["geometric_encoding"])
        geometric_encoding_out = tensor_model.predict(x_test)
        print("geometric_encoding out shape: {}".format(geometric_encoding_out.shape))

        ### Decide on number of digit samples to take
        n_digit_samples = 5

        ### Find indices for required number of digit samples
        digit_list = []
        for digit in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            index_list = np.where(y_test == digit)
            digit_list.append(index_list[0][:n_digit_samples])

        ### Create default figure and outer grid
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(10, n_digit_samples, wspace=.2, hspace=0)

        ### Loop through the outer figure and create inner figures, which are then populated with graphs
        count = 0
        for index_list in digit_list:
            for index in index_list:
                inner = gridspec.GridSpecFromSubplotSpec(1, n_digit_samples,
                                subplot_spec=outer[count], wspace=0.3, hspace=0.1)

                ax = plt.Subplot(fig, inner[0])

                ax.imshow(np.concatenate((x_test[index,:,:,0], predictions[index,:,:,0]), axis=1), cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

                ax = plt.Subplot(fig, inner[1])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(geometric_encoding_out[index,:,:,0], cmap="gray")
                fig.add_subplot(ax)

                count += 1

        ### Save graph picture
        save_file_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name
        save_file_path += "/mnist_samples/"
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        save_file_path += "sample-0.png"

        fig.savefig(save_file_path, dpi=400)
















