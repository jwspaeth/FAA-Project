
import os
import time

from tensorflow.keras.callbacks import Callback

class MyRecordingCallback(Callback):

    def __init__(self, master_config):
        self.data_config = master_config.Core_Config.Data_Config
        self.save_config = master_config.Core_Config.Save_Config
        self.model_config = master_config.Core_Config.Model_Config

        self.child_name = master_config.child_name

        self.epoch_count = 0
        self.recent_epoch_logs = 0
        self.epoch_start_time = 0

    def on_train_begin(self, logs):
        print("Training begin!", flush=True)

    def on_epoch_begin(self, epoch, logs):
        print()
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs):

        self.recent_epoch_logs = self._create_log(logs)
        if self.epoch_count % self.save_config.Output.checkpoint_trigger == 0:
            self._checkpoint()

        self.epoch_count += 1

        total_epoch_time = time.time() - self.epoch_start_time
        print("Total epoch time: {} secs".format(total_epoch_time), flush=True)

    def on_train_end(self, logs):

        print("Training finished!", flush=True)

        self._checkpoint()

    def on_test_batch_end(self, batch, logs):

        self.recent_epoch_logs = self._create_log(logs)

        self.on_train_end(logs)

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





