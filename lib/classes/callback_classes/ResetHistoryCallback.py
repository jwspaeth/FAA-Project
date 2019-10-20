
import resource

import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import Callback

class ResetHistoryCallback(Callback):
    '''
    Reset History Callback: resets history every epoch to prevent memory buildup
    '''

    def __init__(self, *args):
        pass

    def on_epoch_end(self, epoch, logs):

        self.model.history.history.clear()
        self.model.history.epoch.clear()