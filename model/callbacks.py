from keras.callbacks import Callback
from sklearn.metrics import f1_score, classification_report
from keras.utils import to_categorical
import numpy as np


class F1score(Callback):

    def __init__(self, data_generator, steps_per_epoch):
        super(F1score, self).__init__()
        self.steps = steps_per_epoch
        self.data_generator = data_generator
        #self.p = preprocessor

    def on_epoch_end(self, epoch, logs={}):
        accs = []
        label_true = []
        label_pred = []
        for i in range(self.steps):
            #try:
            x_true, y_true,  sequence_lengths = next(self.data_generator)
            y_pred = self.model.predict_on_batch(x_true)

            for lab, lab_pred, length in zip(y_true, y_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]

                lab = np.argmax(lab, axis=1)
                lab_pred = np.argmax(lab_pred, axis=1)
                accs += [a==b for (a, b) in zip(lab, lab_pred)]


                label_true.extend(lab)
                label_pred.extend(lab_pred)
                #break
            #break
            #except StopIteration:
                #break

        #print(label_true)
        label_true = np.asarray(label_true)
        print("Truths: ", label_true)
        #print(label_pred)
        label_pred = np.asarray(label_pred)
        print("Preds: ", label_pred)

        acc = np.mean(accs)

        micro_score = f1_score(label_true, label_pred, average='micro')
        print("acc: ", 100*acc)

        micro_score = f1_score(label_true, label_pred, average='micro')
        print(' - micro f1: {:04.2f}'.format(micro_score * 100))

        macro_score = f1_score(label_true, label_pred, average='macro')
        print(' - macro f1: {:04.2f}'.format(macro_score * 100))

        weighted_score = f1_score(label_true, label_pred, average='weighted')
        print(' - weighted f1: {:04.2f}'.format(weighted_score * 100))
        #print(classification_report(label_true, label_pred))
        logs['f1'] = [micro_score, macro_score, weighted_score]
