
import resource

import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import Callback

class MemoryCallback(Callback):
    '''
    Memory Callback: logs memory during training
    '''

    def __init__(self, *args):
        pass

    def on_train_begin(self, logs):

        self.train_begin_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def on_epoch_begin(self, epoch, logs):

        self.epoch_begin_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Epoch begin - memory usage: {}".format(self.epoch_begin_usage))

    def on_epoch_end(self, epoch, logs):

        epoch_end_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        print("Epoch end - memory usage: {}".format(epoch_end_usage))
        epoch_total_usage = epoch_end_usage - self.epoch_begin_usage
        print("Epoch change in memory usage: {}".format(epoch_total_usage))

    def on_train_end(self, logs):

        train_end_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        train_total_usage = train_end_usage - self.train_begin_usage

        print("Total memory increase: {}".format(train_total_usage))
