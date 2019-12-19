
import os
import time

from tensorflow.keras.callbacks import Callback

class MyRecordingCallback(Callback):

    def __init__(self, master_config):
        self.iteration_parameters = master_config.iteration_parameters

        self.data_config = master_config.Core_Config.Data_Config
        self.save_config = master_config.Core_Config.Save_Config
        self.model_config = master_config.Core_Config.Model_Config

        self.child_name = master_config.child_name

        self.epoch_count = 1
        self.recent_epoch_logs = 0
        self.epoch_start_time = 0

        self.best_val_epoch = 0
        self.best_val_value = None

    def on_train_begin(self, logs):
        print("Training begin!", flush=True)

    def on_epoch_begin(self, epoch, logs):
        print()
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs):

        if "val_loss" in logs.keys():
            print("val_loss: {}".format(logs["val_loss"]))
            if self.best_val_value is None:
                self.best_val_value = logs["val_loss"]
                self.best_val_epoch = self.epoch_count
            elif logs["val_loss"] <= self.best_val_value:
                self.best_val_value = logs["val_loss"]
                self.best_val_epoch = self.epoch_count
        else:
            print("no val_loss")
        self.recent_epoch_logs = self._create_log(logs)
        if self.epoch_count % self.save_config.Output.checkpoint_trigger == 0:
            self._checkpoint()

        self.epoch_count += 1

        total_epoch_time = time.time() - self.epoch_start_time
        self._create_statistics_file()
        print("Total epoch time: {} secs".format(total_epoch_time), flush=True)

    def on_train_end(self, logs):

        print("Training finished!", flush=True)

        self._checkpoint()

    '''def on_test_end(self, logs):

        print("test end logs: {}".format(logs.keys()))
        self.recent_epoch_logs = self._create_log(logs)

        self.on_train_end(logs)'''

    def _create_statistics_file(self):
        
        ### Create statistics representative of training
        ### This includes printing total loss and all separate regularization losses
        print("Saving statistics...", flush=True)
        save_file_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name
        save_file_path += "/statistics.txt"

        total_loss = self.recent_epoch_logs["loss"]

        with open(save_file_path, "w") as f:
            f.write("Experiment parameters: {}\n".format(self.iteration_parameters))
            f.write("Epoch count: {}\n".format(self.epoch_count))
            f.write("Train loss: {}\n".format(round(total_loss, 5)))
            f.write("Validation loss: {}\n".format(round(self.recent_epoch_logs["val_loss"], 5)))
            if self.best_val_value is not None:
                f.write("Best validation: epoch {}, loss {}\n".format(self.best_val_epoch, round(self.best_val_value, 5)))

        print("Statistics saved!", flush=True)

    def _checkpoint(self):
        print("Checkpointing...", flush=True)
        self._save_model()
        print("Finished checkpointing!", flush=True)

    def _save_model(self):
        print("Saving model...", flush=True)
        save_folder_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name
        save_folder_path += "/model_checkpoints/"

        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        self.model.save_weights(save_folder_path + "model_weights_epoch_{}.h5".format(self.epoch_count))
        print("Model saved!", flush=True)

    def _create_log(self, logs):

        return logs





