from keras.callbacks import Callback
from sklearn.metrics import f1_score, classification_report
from keras.utils import to_categorical
import numpy as np


class F1score(Callback):

    def __init__(self, data_generator, steps_per_epoch, config, preprocessor=None):
        super(F1score, self).__init__()
        self.steps = steps_per_epoch
        self.data_generator = data_generator
        self.config=config
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


        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        for i in range(self.steps):
            x_true, y_true = next(self.data_generator)
            sequence_lengths = next(self.seq_length_generator)
            y_pred = self.model.predict_on_batch(x_true)
            for lab, lab_pred, length in zip(y_true, y_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        print(' - f1: {:04.2f}'.format(f1 * 100))
        logs['f1'] = f1
        """
        """
        label_true = []
        label_pred = []
        for i in range(self.steps):
            x_true, y_true = next(self.generator)
            lengths = x_true[-1]
            y_pred = self.model.predict_on_batch(x_true)



            label_true.extend(np.argmax(y_true, axis=2))
            label_pred.extend(np.argmax(y_pred, axis=2))
            break
        print(label_true)
        label_true = np.asarray(label_true).flatten()
        print(label_true)
        print(label_pred)
        label_pred = np.asarray(label_pred).flatten()
        print(label_pred)


        score = f1_score(label_true, label_pred, average='micro')
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(label_true, label_pred))
        logs['f1'] = score"""
