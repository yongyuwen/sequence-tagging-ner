from keras.callbacks import Callback
from sklearn.metrics import f1_score, classification_report
from keras.utils import to_categorical
import numpy as np


class F1score(Callback):

    def __init__(self, data_generator, steps_per_epoch, run_evaluate):
        super(F1score, self).__init__()
        self.steps = steps_per_epoch
        self.data_generator = data_generator
        self.run_evaluate = run_evaluate

    def on_epoch_end(self, epoch, logs={}):
        (micro_score, macro_score, weighted_score) = self.run_evaluate(self.data_generator, self.steps)
        logs['f1'] = [micro_score, macro_score, weighted_score]

class LossHistory(Callback):

    def __init__(self, step_decay):
        self.step_decay = step_decay

    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []

    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(self.step_decay(len(self.losses)))
